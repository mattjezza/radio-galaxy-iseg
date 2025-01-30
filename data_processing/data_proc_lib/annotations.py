import json
import os

import numpy as np
import supervision as sv
from PIL import Image


def create_transformed_anno(transformed, original):
    """
    Create a new annotation for the transformed image and associated bounding boxes,
    masks, classes and keypoints.

    :param transformed: A dictionary of transformed data for exactly one annotation
    :param original: Exactly one original (not transformed) annotation.
    :return: A single annotation for one image.
    """
    anno = {}
    segmentations = []

    # Create polygons from transformed masks
    polygons = sv.mask_to_polygons(transformed["masks"])
    for p in polygons:
        segmentations.append([float(s) for s in p.flatten()])

    anno["segmentation"] = segmentations

    transformed_bbox_area = transformed["bboxes"][2] * transformed["bboxes"][3]

    anno["num_keypoints"] = original["num_keypoints"]
    anno["area"] = transformed_bbox_area
    anno["iscrowd"] = original["iscrowd"]
    anno["bbox"] = list(transformed["bboxes"])
    if "keypoints" in transformed:
        anno["keypoints"] = [int(s) for s in transformed["keypoints"]]
    anno["category_id"] = transformed["bbox_classes"]

    return anno


def data_skeleton(annos):
    """
    Read annotation file and return a skeleton to build a new annotation file.
    :param annos: Source annotation file
    :return: Dictionary containing skeleton annotation information
    """

    with open(annos, "r") as myfile:
        data = myfile.read()

    json_data = json.loads(data)
    data_skeleton = {
        "info": json_data["info"],
        "licenses": json_data["licenses"],
        "images": [],
        "categories": json_data["categories"],
        "annotations": [],
    }

    return data_skeleton


def get_next_id(aug_data, aidx):
    """
    Get the next id for annotation/image.
    """
    if len(aug_data["annotations"]) > 0:
        id = aug_data["annotations"][-1]["id"] + 1
    else:
        id = 0 + aidx

    return id


def next_anno_id(data, first_anno_id, anno_idx=0):
    """
    Get the next annotation id.
    """
    if len(data["annotations"]) > 0:
        anno_id = data["annotations"][-1]["id"] + anno_idx + 1
    else:
        anno_id = first_anno_id + anno_idx + 1

    return anno_id


def next_image_id(data, first_image_id):
    """
    Get the next image id.
    """
    if len(data["images"]) > 0:
        image_id = data["images"][-1]["id"] + 1
    else:
        image_id = first_image_id + 1

    return image_id


def all_data_for_image(image_id, dir, cocoidx):
    """
    Retrieve the image, image file name and annotations for an image
    :param image_id: An image_id
    :param dir: Path to find the "train" directory containing images
    :param cocoidx: A coco index for this dataset
    :return: image as a numpy array, image file name and annotations for an image
    """
    image_file_name = cocoidx.loadImgs(image_id)[0]["file_name"]
    image_path = os.path.join(dir, image_file_name)
    image = np.array(Image.open(image_path))
    annotation_ids = cocoidx.getAnnIds(imgIds=image_id)
    annotations = cocoidx.loadAnns(annotation_ids)

    return image, image_file_name, annotations


def format_img_and_anno_data(annotations, image, cocoidx):
    """
    Several functions use a specific format to present image an annotation data.
    This format is a list of dicts. Each list item corresponds to one annotation.
    There can be several annotations per image (one for each galaxy).
    Each dict contains the individual annotation data fields.
    :param annotations: Annotations for this image.
    :param image: The image as a numpy array.
    :param cocoidx: The coco index.
    :return: The image and annotation data, formatted as a list of dicts.
    """
    img_and_annos = []
    for i in range(len(annotations)):

        transform_dict = {
            "image": image,
            "masks": cocoidx.annToMask(annotations[i]),
            "bboxes": annotations[i]["bbox"],
            "bbox_classes": annotations[i]["category_id"],
        }
        if "keypoints" in annotations[i]:
            transform_dict["keypoints"] = annotations[i]["keypoints"]

        img_and_annos.append(transform_dict)
    return img_and_annos
