import json
import os
import random

import albumentations as A
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from numpy.ma.testutils import assert_equal
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

SCALED_IMAGE_SIZE = 32

R_CHANNEL_CUTOUT_SIZE = 10

import sys

sys.path.insert(1, "../..")

from .annotations import create_transformed_anno, data_skeleton
from .cleaning import find_invalid_annotations
from .utilities import create_directories, BLACK_PIXEL, IMAGE_HEIGHT, IMAGE_WIDTH


def augment(
    num_augs, output_path, input_path, transform, make_backgrounds=True, balance=False
):
    """
    Perform offline augmentation on a dataset.
    :param num_augs: Number of times to loop through the dataset applying the transform.
    :param output_path: Path to the place the augmented output files.
    :param input_path: Path to the input files to augment.
    :param transform: Transform to apply.
    :param make_backgrounds: If True, generate background images.
    :param balance: If True, oversample the under-represented classes to balance the dataset.
    :return: None.
    """
    # Count the number of background images
    backgrounds = 0

    create_directories(output_path)
    output_image_path = os.path.join(output_path, "train")
    train_annos = os.path.join(input_path, "annotations", "train.json")

    # aug_train_data is all the data we'll eventually write to the transformed annotations file
    aug_train_data = data_skeleton(train_annos)

    with open(train_annos, "r") as myfile:
        data = myfile.read()
    json_data = json.loads(data)

    # Find the max value of the image id and annotation id in the original data
    max_train_image_id = json_data["images"][-1]["id"]
    max_train_anno_id = json_data["annotations"][-1]["id"]

    coco = COCO(train_annos)
    image_ids = coco.getImgIds()
    exclude = list(find_invalid_annotations(coco))

    image_file_names = [
        coco.loadImgs(image_id)[0]["file_name"] for image_id in image_ids
    ]

    # image_aug_count is a dict to count the number of times an original image is transformed
    # We use this as a prefix to the augmented file name.
    zero_counts = [0 for im in image_file_names]
    image_aug_count = {k: v for k, v in zip(image_file_names, zero_counts)}

    # Loop through all original images num_augs times
    for augnum in range(num_augs):
        for idx, image_id in enumerate([i for i in image_ids if i not in exclude]):

            # Whether this example will be turned into a background image
            background = False

            # aug_annos is a list of transformed annotations corresponding to one transformed image
            aug_annos = []

            image_file_name = coco.loadImgs(image_id)[0]["file_name"]
            image_path = os.path.join(input_path, "train", image_file_name)
            image_aug_count[image_file_name] += 1

            aug_file_name = (
                str(image_aug_count[image_file_name]) + "_" + image_file_name
            )
            output_image_file_path = os.path.join(output_image_path, aug_file_name)

            # Read the image and associated annotations
            image = np.array(Image.open(image_path))
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            annotations = coco.loadAnns(annotation_ids)
            # print(annotations)

            # In case a transform has shifted all galaxies out of an image and no annotations remain.
            if len(annotations) == 0:
                continue

            # Random chance of an image containing exactly one galaxy to be
            # converted to a background.
            if (
                make_backgrounds is True
                and len(annotations) == 1
                and random.choice(range(10)) == 0
            ):
                background = True
                image = do_background_transform(annotations, image, coco)
                if not np.any(image):
                    continue
                else:
                    backgrounds += 1

            # Balance classes by only selecting FR-II or R galaxies a fraction of the time they appear in the dataset
            if balance:
                if annotations[0]["category_id"] == 1:
                    if random.choice(range(4)) > 0:
                        continue
                if annotations[0]["category_id"] == 4:
                    if random.choice(range(4)) > 2:
                        continue

            # Perform the transform. Output is a list of dicts.
            transformed_img_and_annos = do_transform(
                annotations, image, transform, coco
            )  # area_counts) #, exit_code, area_counts

            # It's possible that the transformation has introduced artefacts, in which we discard it.
            if len(transformed_img_and_annos) == 0:
                # artefacts[exit_code] += 1
                # print("Rejected due to artefacts.")
                continue

            # Image data in the annotations file for this image.
            aug_image_data = {
                "file_name": aug_file_name,
                "height": transformed_img_and_annos[0]["image"].shape[0],
                "width": transformed_img_and_annos[0]["image"].shape[1],
                "id": max_train_image_id * (augnum + 1) + idx + 1,
            }

            transformed_image = transformed_img_and_annos[0]["image"]

            # Backgrounds have no annotations. For non-backgrounds, create the transformed annotation.
            if background is False:
                for a, transformed in enumerate(transformed_img_and_annos):

                    anno = create_transformed_anno(transformed, annotations[a])
                    anno["image_id"] = aug_image_data["id"]

                    if len(aug_train_data["annotations"]) > 0:
                        anno["id"] = aug_train_data["annotations"][-1]["id"] + a + 1
                    else:
                        anno["id"] = max_train_anno_id + a + 1

                    aug_annos.append(anno)

            # Write the transformed image to a png file.
            Image.fromarray(transformed_image).save(output_image_file_path)
            # print("Saving image")

            # Append the image-related data to the annotation.
            aug_train_data["images"].append(aug_image_data)

            # There can be multiple annotation files per image.
            # Write them all to the annotation for this image.
            for aug in aug_annos:
                aug_train_data["annotations"].append(aug)

    # Write the annotations file.
    train_annos_filepath = os.path.join(output_path, "annotations/train.json")

    with open(train_annos_filepath, "w") as f:
        json.dump(aug_train_data, f, ensure_ascii=False, indent=4)

    print(f"Completed! Augmented data is in {output_path}.")
    print(f"Number of background images: {backgrounds}")


def do_transform(annotations, image, transform, coco):
    """
    Perform the transformation on image and annotation data. If the image
    has multiple annotations, which is the case if an image contains
    multiple objects of interest, then perform the same transformation
    on all the annotations associated with the image. Transformations are
    applied to the image, bounding boxes, masks, classes and keypoints.

    :param annotations: A list of annotations for this image.
    :param image: Image as an RGB ndarray.
    :param transform: The Albumentations transform to perform.
    :param coco: COCO index for the dataset.
    :return: A list of dicts. Each dict contains one transformed image, one mask,
    one bounding box, one class and one keypoint. Each dict in the list corresponds
    to a single annotation. Each dict corresponds to the same image. Overall,
    the list contains all the transformed annotations corresponding to a single image.
    """

    transformed_img_and_annos = []

    # We pass the Albumentations transform function a series of lists.
    # Each list contains one type of annotation data.
    # The length of each list is the same, and is equal to the number of
    # annotations corresponding to this image.
    masks = []
    bboxes = []
    bbox_classes = []
    keypoints = []

    for i in range(len(annotations)):
        # print(f"i: {i}, annotations: {annotations[i]}")
        if annotations[i]["area"] == 0:
            print("Rejected due to zero area.")
            return []
        if not annotations[i]["segmentation"]:
            print("Rejected due to no segmentation.")
            return []
        masks.append(coco.annToMask(annotations[i]))
        bboxes.append(annotations[i]["bbox"])
        bbox_classes.append(annotations[i]["category_id"])
        if "keypoints" in annotations[i]:
            keypoints.append(annotations[i]["keypoints"])

    # print(bboxes)
    # print(bbox_classes)

    transformed = transform(
        image=image,
        masks=masks,
        bboxes=bboxes,
        bbox_classes=bbox_classes,
        keypoints=keypoints,
    )

    # print(transformed["keypoints"])

    # If no transforms were applied break and go to the next image
    if not any([t["applied"] for t in transformed["replay"]["transforms"]]):
        print("No transform applied")
        return []

    # Check if the transformation introduced artefacts or the galaxies overflowed the boundary
    artefacts = transform_has_artefacts(
        transformed,
        annotations,
        coco,
        len(masks),
        len(bboxes),
        len(bbox_classes),
        len(keypoints),
    )
    if artefacts is True:
        # print("Artefacts introduced")
        return []

    for a in range(len(annotations)):

        # Create tight bounding boxes from mask
        tight_bbox_xyxy = sv.mask_to_xyxy(
            transformed["masks"][a].reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT])
        )
        width = int(tight_bbox_xyxy[0][2] - tight_bbox_xyxy[0][0])
        height = int(tight_bbox_xyxy[0][3] - tight_bbox_xyxy[0][1])

        tight_bbox_xywh = [
            int(tight_bbox_xyxy[0][0]),
            int(tight_bbox_xyxy[0][1]),
            width,
            height,
        ]

        # There is an annotation but the components (bbox, mask etc) are all empty.
        if tight_bbox_xywh == [0, 0, 0, 0]:
            print("Rejected due to zero area after tight bounding box creation.")
            # print(f"annotations: {annotations[a]}")
            return []

        # The output format is a list of dicts, one list item per annotation.
        # Each dict entry corresponds to one of the transformed annotation fields.
        transform_dict = {
            "image": transformed["image"],
            "masks": transformed["masks"][a],
            # "bboxes": transformed["bboxes"][a],  # tight_bbox_xywh,
            "bboxes": tight_bbox_xywh,
            "bbox_classes": transformed["bbox_classes"][a],
        }
        if len(transformed["keypoints"]) > 0:
            transform_dict["keypoints"] = transformed["keypoints"][a]

        transformed_img_and_annos.append(transform_dict)

    return transformed_img_and_annos


def do_background_transform(annotations, image, coco):
    """
    Create a background image. This is an image that contains no objects of
    interest.

    This function takes an image and annotations and removes all objects of
    interest from it. For simplicity, this function expects that there will
    be exactly one annotation associated with the image, i.e. just one
    object of interest in the image. It's the caller's responsibility to
    ensure this.

    :param annotations:
    :param image:
    :param coco:
    :return:
    """
    mask = coco.annToMask(annotations[0])

    background_transform = A.Compose([A.MaskDropout(p=1)])

    transformed = background_transform(image=image, mask=mask)

    if np.any(transformed["mask"]):
        return []

    return transformed["image"]


def transform_has_artefacts(
    transformed,
    annotations,
    coco,
    num_masks,
    num_bboxes,
    num_bbox_classes,
    num_keypoints,
):
    """
    Check if the transformed image has any artefacts.

    :param transformed:
    :param annotations:
    :param coco:
    :param num_masks:
    :param num_bboxes:
    :param num_bbox_classes:
    :param num_keypoints:
    :return:
    """
    artefacts = False

    # Check if the number of masks, bboxes, bbbox_classes or keypoints changed.
    # We don't check the number of polygons. It was found that the number of polygons
    # could change in the transform but this did not significantly affect the quality
    # of the output mask. If a serious problem was introduced, it will be caught by
    # the check on mask area later on.
    if len(transformed["masks"]) != num_masks:
        artefacts = True
        print("Rejected due to different number of masks.")
    if len(transformed["masks"]) == 0:
        print("Rejected due to empty mask.")
        artefacts = True
    if len(transformed["bboxes"]) != num_bboxes:
        print("Rejected due to different number of bboxes.")
        print(f"{len(transformed['bboxes'])}, {num_bboxes}")
        artefacts = True
    if len(transformed["bbox_classes"]) != num_bbox_classes:
        print("Rejected due to different number of bbox classes.")
        print(f"{len(transformed['bbox_classes'])}, {num_bbox_classes}")
        artefacts = True
    if len(transformed["keypoints"]) != num_keypoints:
        print("Rejected due to different number of keypoints.")
        artefacts = True

    for a in range(len(annotations)):

        # Check mask area hasn't changed significantly.
        rle = coco_mask.encode(np.asfortranarray(transformed["masks"][a]))
        transformed_area = coco_mask.area(rle)
        untransformed_mask = coco.annToMask(annotations[a])
        untransformed_rle = coco_mask.encode(np.asfortranarray(untransformed_mask))
        untransformed_area = coco_mask.area(untransformed_rle)

        # If mask area changes by more than 5 pixels, or by more than 5% of the
        # original mask size (whichever is greater), then reject the image.
        if abs(int(transformed_area) - int(untransformed_area)) > 5:
            if (
                transformed_area > 1.05 * untransformed_area
                or transformed_area < 0.95 * untransformed_area
            ):
                artefacts = True
                print("Rejected due to different mask area.")
                break

    return artefacts


def do_cutout(annotation, image, coco):
    """
    Create a cutout of an individual galaxy in an image.
    :param annotation: A single annotation corresponding to one galaxy
    :param image: numpy array representing the complete
    :param coco: coco index for this data
    :return: A list of dicts. Each dict contains one transformed image, one mask,
    one bounding box, one class and one keypoint. Each dict in the list corresponds
    to a single annotation. Each dict corresponds to the same image. Overall,
    the list contains all the transformed annotations corresponding to a single image.
    """
    transformed_img_and_annos = []

    mask = coco.annToMask(annotation)
    bboxes = annotation["bbox"]
    bbox_classes = annotation["category_id"]
    keypoints = annotation["keypoints"]

    # R channel of output image.
    rchannel = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    # kp_ymax, kp_ymin, kp_xmax, kp_xmin define the region to extract from the
    # original image's R channel. We want this to be R_CHANNEL_CUTOUT_SIZE / 2
    # square, centered on the keypoint. If the keypoint is too close to the
    # edge of the image to allow this we reduce the cutout size in the R channel
    # to fit.
    if keypoints[1] + (R_CHANNEL_CUTOUT_SIZE // 2) < IMAGE_HEIGHT:
        kp_ymax = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_ymax = IMAGE_HEIGHT - 1 - keypoints[1]
    if keypoints[1] - (R_CHANNEL_CUTOUT_SIZE // 2) >= 0:
        kp_ymin = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_ymin = keypoints[1]
    if keypoints[0] + (R_CHANNEL_CUTOUT_SIZE // 2) < IMAGE_WIDTH:
        kp_xmax = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_xmax = IMAGE_WIDTH - 1 - keypoints[0]
    if keypoints[0] - (R_CHANNEL_CUTOUT_SIZE // 2) >= 0:
        kp_xmin = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_xmin = keypoints[0]

    # Select the section to be cut out of the R channel of the original image and reshape
    cutout = [
        image[i, j, 0]
        for i in range(keypoints[1] - kp_ymin, keypoints[1] + kp_ymax)
        for j in range(keypoints[0] - kp_xmin, keypoints[0] + kp_xmax)
    ]
    cut_array = np.reshape(cutout, (kp_ymin + kp_ymax, kp_xmin + kp_xmax))

    # Use the cut_array above to set the values of the R channel at the same coordinates
    rchannel[
        keypoints[1] - kp_ymin : keypoints[1] + kp_ymax,
        keypoints[0] - kp_xmin : keypoints[0] + kp_xmax,
    ] = cut_array

    # Convert the mask into a binary mask (i.e. True/False values)
    binary_mask = np.logical_and(mask, mask)
    # Copy the parts of G and B channels of original image that are in the binary mask
    gchannel = np.multiply(binary_mask, image[:, :, 1])
    bchannel = np.multiply(binary_mask, image[:, :, 2])

    # Stack all three channels
    transformed_image = np.dstack([rchannel, gchannel, bchannel]).astype(np.uint8)

    # transform = A.GaussianBlur(blur_limit=3, p=1)
    # transformed = transform(image=transformed_image)
    # transformed_image = transformed["image"]

    # Format as a list of dicts. In this case, the list will only have one element
    # because this is just one galaxy, but the rest of the code expects to receive a list
    transformed_img_and_annos.append(
        {
            "image": transformed_image,
            "masks": mask,
            "bboxes": bboxes,
            "bbox_classes": bbox_classes,
            "keypoints": keypoints,
        }
    )

    return transformed_img_and_annos


def do_square_cutout(annotation, image, coco):
    """
    Create a cutout of an individual galaxy in an image and scale it to a standard size.

    :param annotation: A single annotation corresponding to one galaxy
    :param image: numpy array representing the complete
    :param coco: coco index for this data
    :return: A list of dicts. Each dict contains one transformed image, one mask,
    one bounding box, one class and one keypoint. Each dict in the list corresponds
    to a single annotation. Each dict corresponds to the same image. Overall,
    the list contains all the transformed annotations corresponding to a single image.
    """
    transformed_img_and_annos = []

    mask = coco.annToMask(annotation)
    bboxes = annotation["bbox"]
    bbox_classes = annotation["category_id"]
    keypoints = annotation["keypoints"]

    # R channel of output image.
    rchannel = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

    # kp_ymax, kp_ymin, kp_xmax, kp_xmin define the region to extract from the
    # original image's R channel. We want this to be R_CHANNEL_CUTOUT_SIZE / 2
    # square, centered on the keypoint. If the keypoint is too close to the
    # edge of the image to allow this we reduce the cutout size in the R channel
    # to fit.
    if keypoints[1] + (R_CHANNEL_CUTOUT_SIZE // 2) < IMAGE_HEIGHT:
        kp_ymax = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_ymax = IMAGE_HEIGHT - 1 - keypoints[1]
    if keypoints[1] - (R_CHANNEL_CUTOUT_SIZE // 2) >= 0:
        kp_ymin = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_ymin = keypoints[1]
    if keypoints[0] + (R_CHANNEL_CUTOUT_SIZE // 2) < IMAGE_WIDTH:
        kp_xmax = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_xmax = IMAGE_WIDTH - 1 - keypoints[0]
    if keypoints[0] - (R_CHANNEL_CUTOUT_SIZE // 2) >= 0:
        kp_xmin = R_CHANNEL_CUTOUT_SIZE // 2
    else:
        kp_xmin = keypoints[0]

    # Select the section to be cut out of the R channel of the original image and reshape
    cutout = [
        image[i, j, 0]
        for i in range(keypoints[1] - kp_ymin, keypoints[1] + kp_ymax)
        for j in range(keypoints[0] - kp_xmin, keypoints[0] + kp_xmax)
    ]
    cut_array = np.reshape(cutout, (kp_ymin + kp_ymax, kp_xmin + kp_xmax))

    # Use the cut_array above to set the values of the R channel at the same coordinates
    rchannel[
        keypoints[1] - kp_ymin : keypoints[1] + kp_ymax,
        keypoints[0] - kp_xmin : keypoints[0] + kp_xmax,
    ] = cut_array

    # Convert the mask into a binary mask (i.e. True/False values)
    binary_mask = np.logical_and(mask, mask)
    # Copy the parts of G and B channels of original image that are in the binary mask
    gchannel = np.multiply(binary_mask, image[:, :, 1])
    bchannel = np.multiply(binary_mask, image[:, :, 2])

    # Stack all three channels
    transformed_image = np.dstack([rchannel, gchannel, bchannel]).astype(np.uint8)
    # print(transformed_image.shape)

    # Create a bounding box that tightly surrounds the galaxy.
    tight_bbox_xyxy = sv.mask_to_xyxy(mask.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT]))

    # Now embed the galaxy in a square image (SCALED_IMAGE_SIZExSCALED_IMAGE_SIZE pixels)
    square_image = make_square(
        transformed_image[
            tight_bbox_xyxy[0][1] : tight_bbox_xyxy[0][3],
            tight_bbox_xyxy[0][0] : tight_bbox_xyxy[0][2],
            :,
        ],
        BLACK_PIXEL,
        SCALED_IMAGE_SIZE,
    )

    # Actual scaling is done in the calling function.

    # Format as a list of dicts. In this case, the list will only have one element
    # because this is just one galaxy, but the rest of the code expects to receive a list

    # The annotations are not valid in this because they have not been transformed after scaling.
    # We don't need the annotations, though, just the image.
    transformed_img_and_annos.append(
        {
            "image": square_image,
            "masks": mask,
            "bboxes": bboxes,
            "bbox_classes": bbox_classes,
            "keypoints": keypoints,
        }
    )

    return transformed_img_and_annos


def make_square(m, val, scaled_image_size):
    """
    Function to convert an array m into a square array of size (max(m.shape), max(m.shape))

    This function adapted from https://stackoverflow.com/questions/10871220/making-a-matrix-square-and-padding-it-with-desired-value-in-numpy

    :param m: The array to be converted
    :param val: The value to be used to pad the array
    :param scaled_image_size: The size of the output array
    """

    h = m.shape[0]
    w = m.shape[1]

    if h > w:
        padding = ((0, 0), (0, h - w), (0, 0))
    else:
        padding = ((0, w - h), (0, 0), (0, 0))

    square_m = np.pad(m, padding, mode="constant", constant_values=val)

    sq_h = square_m.shape[0]
    sq_w = square_m.shape[1]

    assert_equal(sq_h, sq_w, "make_square(): the matrix is not square")

    # If both sides are greater than or equal to the target scaled size then no padding is needed.
    if sq_h >= scaled_image_size and sq_w >= scaled_image_size:
        return square_m

    # Else, pad the array so that each side is >= scaled_image_size
    if sq_h < scaled_image_size and sq_w < scaled_image_size:
        padding = ((0, scaled_image_size - sq_h), (0, scaled_image_size - sq_w), (0, 0))
    elif sq_w < scaled_image_size:
        padding = ((0, 0), (0, scaled_image_size - sq_w), (0, 0))
    elif sq_h < scaled_image_size:
        padding = ((0, scaled_image_size - sq_h), (0, 0), (0, 0))

    return np.pad(square_m, padding, mode="constant", constant_values=val)


def do_paste(image, object_to_paste, threshold=0):
    """
    Paste an object of interest into an image.
    :param image: The image (numpy array) to use as the base to paste into.
    :param object_to_paste: The cutout image to be pasted into the image
    :return: The image (numpy array) after pasting the object into it.
    """
    for channel in range(object_to_paste.shape[2]):
        image[:, :, channel][np.where(object_to_paste[:, :, channel] > threshold)] = (
            object_to_paste[:, :, channel][
                np.where(object_to_paste[:, :, channel] > threshold)
            ]
        )

    return image


def shift_and_rotate_object():
    """
    Albumentations transform to shift and rotate an image and annotations.
    :return: The Albumentations transform.
    """
    transform = A.ReplayCompose(
        [
            A.Affine(
                translate_percent=(-0.4, 0.4),
                rotate=(-180, 180),
                shear=0.0,
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                keep_ratio=True,
                rotate_method="ellipse",
                p=1,
            )
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return transform


def shift_object():
    """
    Albumentations transform to shift and rotate an image and annotations.
    :return: The Albumentations transform.
    """
    transform = A.ReplayCompose(
        [
            A.Affine(
                translate_percent=(-0.4, 0.4),
                rotate=0,
                shear=0.0,
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                keep_ratio=True,
                rotate_method="ellipse",
                p=1,
            )
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return transform


def rotate():
    """
    Albumentations transform to rotate an image and annotations.
    :return: The Albumentations transform.
    """
    transform = A.ReplayCompose(
        [
            A.Affine(
                rotate=(-180, 180),
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                keep_ratio=True,
                rotate_method="ellipse",
                p=1,
            )
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return transform


def random_rotate_90():
    """
    Albumentations transform to rotate an image and annotations by 0, 90, 180 or 270 degrees.
    :return: The Albumentations transform.
    """
    transform = A.ReplayCompose(
        [A.RandomRotate90(p=1.0)],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return transform


def rotate_exactly_90():
    """
    Albumentations transform to rotate an image and annotations by 90 degrees.
    :return: The Albumentations transform.
    """
    transform = A.ReplayCompose(
        [
            A.Affine(
                rotate=90,
                shear=0.0,
                interpolation=cv2.INTER_CUBIC,
                mask_interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                fit_output=False,
                keep_ratio=True,
                rotate_method="ellipse",
                p=1,
            )
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
        keypoint_params=A.KeypointParams(format="xy"),
    )

    return transform
