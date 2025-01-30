import json
import os
import shutil

from pycocotools.coco import COCO

from .utilities import create_directories, IMAGE_HEIGHT


def find_invalid_annotations(cocoidx):
    """
    Find any examples with invalid annotations
    :param cocoidx: COCO index describing the images and annotations
    :return: A set of image_ids with invalid annotations
    """
    faulty = []
    image_ids = cocoidx.getImgIds()
    for idx, image_id in enumerate(image_ids):
        annotation_ids = cocoidx.getAnnIds(imgIds=image_id)
        annotations = cocoidx.loadAnns(annotation_ids)
        for ann in annotations:

            # Reject any example with bbox, keypoint or mask polygon coordinates outside image boundary
            if any(p >= IMAGE_HEIGHT or p < 0 for p in ann["bbox"]):
                faulty.append(image_id)
            if "keypoints" in ann:
                if any(p >= IMAGE_HEIGHT or p < 0 for p in ann["keypoints"]):
                    faulty.append(image_id)
            if any(
                p >= IMAGE_HEIGHT or p < 0
                for p in [
                    point for seg in annotations[0]["segmentation"] for point in seg
                ]
            ):
                faulty.append(image_id)

            # Reject any example with >4 coordinates per bbox or >3 values per keypoint
            if len(ann["bbox"]) > 4:
                faulty.append(image_id)

            if "keypoints" in ann and len(ann["keypoints"]) > 3:
                faulty.append(image_id)

    return set(faulty)


def clean(input_path, output_path):
    """
    Clean the data in input_path and copy the result to output_path.
    The result is that output_path will contain a copy of the original dataset
    but with any invalid images and associated annotations removed.

    :param input_path: Path to the input dataset.
    :param output_path: Path to store the cleaned output dataset.
    """

    create_directories(output_path, val_and_test_dirs=True)

    for dataset in ["train", "val", "test"]:
        anno = f"{dataset}.json"
        source_annos = os.path.join(input_path, "annotations", anno)
        if not os.path.isfile(os.path.join(input_path, "annotations", anno)):
            print(f"{anno} does not exist, skipping.")
            continue

        source_images_path = os.path.join(input_path, dataset)
        if not os.path.isdir(source_images_path):
            print(f"{dataset} does not exist, skipping.")
            continue

        with open(source_annos, "r") as f:
            data = f.read()

        json_data = json.loads(data)

        cleaned_data = {
            "info": json_data["info"],
            "licenses": json_data["licenses"],
            "images": [],
            "categories": json_data["categories"],
            "annotations": [],
        }

        coco = COCO(source_annos)
        faulty_image_ids = find_invalid_annotations(coco)

        for im in json_data["images"]:
            if im["id"] in faulty_image_ids:
                continue

            image_path = os.path.join(source_images_path, im["file_name"])
            shutil.copy2(image_path, os.path.join(output_path, dataset))
            cleaned_data["images"].append(im)
            annotation_ids = coco.getAnnIds(imgIds=im["id"])
            annotations = coco.loadAnns(annotation_ids)
            cleaned_data["annotations"].extend(annotations)

        with open(os.path.join(output_path, "annotations", anno), "w") as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=4)
