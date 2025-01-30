import json
import os
import random
import sys
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from numpy.random import choice
from pycocotools.coco import COCO

sys.path.insert(1, "../..")

from .transforms import (
    do_transform,
    shift_object,
    do_paste,
    do_cutout,
    do_square_cutout,
    SCALED_IMAGE_SIZE,
)
from .annotations import (
    create_transformed_anno,
    all_data_for_image,
    format_img_and_anno_data,
    data_skeleton,
    next_anno_id,
    next_image_id,
)
from .cleaning import find_invalid_annotations
from .utilities import create_directories, BLACK_PIXEL
from .utilities import NUM_CATEGORIES


def create_cutouts(input_path, output_path, scale=False):
    """
    Create cutout galaxy images.
    :param input_path: Path to the input data.
    :param output_path: Path to place the cutout images.
    :param scale: Whether to scale the cutout image to SCALED_IMAGE_SIZE
    :return: None
    """
    create_directories(output_path, split_by_class=True)

    train_annos = os.path.join(input_path, "annotations", "train.json")

    # aug_train_data is all the data we'll eventually write to the transformed annotations file
    aug_train_data = data_skeleton(train_annos)

    # all_aug_train_data is a list holding aug_train_data for each individual galaxy category
    all_aug_train_data = []
    for cat in range(NUM_CATEGORIES):
        all_aug_train_data.append(deepcopy(aug_train_data))

    coco = COCO(train_annos)
    image_ids = coco.getImgIds()
    exclude = list(find_invalid_annotations(coco))

    # Loop over images
    for idx, image_id in enumerate([i for i in image_ids if i not in exclude]):

        image_file_name = coco.loadImgs(image_id)[0]["file_name"]
        image_path = os.path.join(input_path, "train", image_file_name)

        image = np.array(Image.open(image_path))

        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        # Loop over the annotations for a given image.
        # In effect, this is a loop over all the galaxies in an image.
        for aidx, annotation in enumerate(annotations):

            # Category of the galaxy. Used to separate the cutouts into categories.
            catid = annotation["category_id"] - 1
            output_image_path = os.path.join(output_path, str(catid), "train")

            if len(all_aug_train_data[catid]["annotations"]) > 0:
                id = all_aug_train_data[catid]["annotations"][-1]["id"] + 1
            else:
                id = 0 + aidx

            aug_file_name = str(f"{aidx + 1}c_{image_file_name}")
            output_image_file_path = os.path.join(output_image_path, aug_file_name)

            if scale:
                transformed_img_and_annos = do_square_cutout(annotation, image, coco)
            else:
                transformed_img_and_annos = do_cutout(annotation, image, coco)

            # Image data in the annotations file for this image.
            aug_image_data = {
                "file_name": aug_file_name,
                "height": transformed_img_and_annos[0]["image"].shape[0],
                "width": transformed_img_and_annos[0]["image"].shape[1],
                "id": id,
            }

            transformed_image = transformed_img_and_annos[0]["image"]

            anno = create_transformed_anno(transformed_img_and_annos[0], annotation)
            anno["image_id"] = aug_image_data["id"]
            anno["id"] = id
            all_aug_train_data[catid]["annotations"].append(anno)

            # Save image and associated data
            if scale:
                im_to_save = Image.fromarray(transformed_image)
                im_to_save.thumbnail(
                    (SCALED_IMAGE_SIZE, SCALED_IMAGE_SIZE), resample=Resampling.LANCZOS
                )
                im_to_save.save(output_image_file_path)
            else:
                Image.fromarray(transformed_image).save(output_image_file_path)

            all_aug_train_data[catid]["images"].append(aug_image_data)

    # Set the output paths
    all_train_annos_filepaths = []
    for cat in range(NUM_CATEGORIES):
        train_annos_filepath = os.path.join(
            output_path, str(cat), "annotations/train.json"
        )
        all_train_annos_filepaths.append(train_annos_filepath)

    for c, filepath in enumerate(all_train_annos_filepaths):
        with open(filepath, "w") as f:
            json.dump(all_aug_train_data[c], f, ensure_ascii=False, indent=4)


def paste(input_dir, output_dir, cutouts_dir, image_dir):
    """
    Paste a galaxy cutout into a training image and update annotations.
    :param input_dir: Path to cleaned training images to paste into.
    :param output_dir: Path to place the output files and annotations.
    :param cutouts_dir: Path to the directory containing per-category galaxy cutouts.
    :param image_dir: Path to the directory containing per-category training images.
    """

    # Setup
    original_annos = os.path.join(input_dir, "annotations", "train.json")

    coco_cutouts_list = cutouts_cocoidx_per_category(cutouts_dir)

    augmented_data = data_skeleton(original_annos)
    coco_original = COCO(original_annos)
    original_image_ids = coco_original.getImgIds()

    # cutout_image_ids = coco_cutouts.getImgIds()
    cutout_image_ids_list = cutouts_image_ids_per_category(coco_cutouts_list)
    transformed_img_and_annos = []
    output_image_dir = os.path.join(output_dir, "train")
    exclude = list(find_invalid_annotations(coco_original))

    # Transform to apply
    transform = shift_object()

    # For constructing new image file names
    original_image_file_names = [
        coco_original.loadImgs(image_id)[0]["file_name"]
        for image_id in original_image_ids
    ]
    zero_counts = [0 for im in original_image_file_names]
    image_aug_count = {k: v for k, v in zip(original_image_file_names, zero_counts)}

    # For finding the maximum image id and maximum annotation id in the original data so we don't reuse them
    with open(original_annos, "r") as myfile:
        data = myfile.read()
    json_data = json.loads(data)
    max_train_id = json_data["images"][-1]["id"]
    max_train_anno_id = json_data["annotations"][-1]["id"]

    # Loop over all original images
    for image_id in [i for i in original_image_ids if i not in exclude]:

        original_image, original_image_file_name, original_image_annos = (
            all_data_for_image(
                image_id, os.path.join(input_dir, "train"), coco_original
            )
        )
        image_aug_count[original_image_file_name] += 1
        aug_file_name = (
            str(image_aug_count[original_image_file_name])
            + "_"
            + original_image_file_name
        )

        # Choose category to balance the data set across categories.
        cat = choice(
            range(NUM_CATEGORIES), 1, p=[1 / 11, 4 / 11, 4 / 11, 2 / 11]
        ).item()
        cutout_image_id = random.choice(cutout_image_ids_list[cat])
        cutout_image, cutout_image_file_name, cutout_image_annos = all_data_for_image(
            cutout_image_id,
            os.path.join(cutouts_dir, str(cat), image_dir),
            coco_cutouts_list[cat],
        )

        original_img_and_annos = format_img_and_anno_data(
            original_image_annos, original_image, coco_original
        )

        image_id = next_image_id(augmented_data, max_train_id)

        transformed_object = do_transform(
            cutout_image_annos, cutout_image, transform, coco_cutouts_list[cat]
        )

        # The shift and rotate transform above may shift the object out of the image frame
        # If so, go to the next one.
        if len(transformed_object) == 0:
            continue

        # Check for overlap
        if check_overlap(transformed_object, original_img_and_annos):
            continue

        transformed_image = do_paste(original_image, transformed_object[0]["image"])
        transformed_img_and_annos = transformed_object
        transformed_img_and_annos.extend(original_img_and_annos)
        for i in range(len(transformed_img_and_annos)):
            transformed_img_and_annos[i]["image"] = transformed_image

        aug_image_data = {
            "file_name": aug_file_name,
            "height": transformed_img_and_annos[0]["image"].shape[0],
            "width": transformed_img_and_annos[0]["image"].shape[1],
            "id": image_id,
        }

        # Save the transformed image, append image data to annotation data list
        transformed_image = transformed_img_and_annos[0]["image"]
        output_path = os.path.join(output_image_dir, aug_file_name)
        Image.fromarray(transformed_image).save(output_path)
        augmented_data["images"].append(aug_image_data)

        # We are adding just one galaxy to each original image.
        # Adding more galaxies can be achieved by repeating the process again.
        # Create the annotation for the newly-added object.
        anno = create_transformed_anno(
            transformed_img_and_annos[0], cutout_image_annos[0]
        )
        anno["image_id"] = aug_image_data["id"]
        anno["id"] = next_anno_id(augmented_data, max_train_anno_id)
        augmented_data["annotations"].append(anno)

        # Create the annotation for the original image. These are after the new annotation
        # in transformed_img_and_annos, hence we start at index 1 instead of 0.
        for a, transformed in enumerate(transformed_img_and_annos[1:]):
            anno = create_transformed_anno(transformed, original_image_annos[a])
            anno["image_id"] = aug_image_data["id"]
            anno["id"] = next_anno_id(augmented_data, max_train_anno_id)
            augmented_data["annotations"].append(anno)

    filepath = os.path.join(output_dir, "annotations/train.json")
    with open(filepath, "w") as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=4)


def check_overlap(object_annos, image_annos):
    """
    Check if any masks in object_annos and image_annos overlap.
    """
    object_mask = object_annos[0]["masks"]
    for a in image_annos:
        if mask_overlap(object_mask, a["masks"]):
            return True

    return False


def mask_overlap(mask1, mask2):
    """Return True if masks overlap"""
    return np.any(cv2.bitwise_and(mask1, mask2))


def cutouts_image_ids_per_category(coco_cutouts_list):
    """
    Create a list containing per-category image ids.
    :param coco_cutouts_list: A list of coco indices (one per category)
    :return: A list containing per-category image ids.
    """
    cutouts_image_ids_list = []
    for cat in range(NUM_CATEGORIES):
        cutouts_image_ids_list.append(coco_cutouts_list[cat].getImgIds())

    return cutouts_image_ids_list


def cutouts_cocoidx_per_category(cutouts_dir):
    """
    Create a list containing per-category coco indices.
    :param cutouts_dir: Path to the cutouts directory
    :return: A list containing per-category coco indices.
    """
    cutouts_cocoidx_list = []
    for cat in range(NUM_CATEGORIES):
        cutouts_cocoidx_list.append(
            COCO(os.path.join(cutouts_dir, str(cat), "annotations", "train.json"))
        )

    return cutouts_cocoidx_list
