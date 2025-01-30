import os
import random
import sys
from copy import deepcopy

import albumentations as A
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

sys.path.insert(1, "../..")
from .utilities import BLACK_PIXEL, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CATEGORIES


def create_pixel_distributions(cutouts_path):
    """
    Create pixel distributions for each cutout galaxy.
    :param cutouts_path: Path to the cutout images
    :return:
    """
    for cat in range(NUM_CATEGORIES):

        train_annos = os.path.join(cutouts_path, str(cat), "annotations", "train.json")
        cutout_image_path = os.path.join(cutouts_path, str(cat), "train")
        coco = COCO(train_annos)
        image_ids = coco.getImgIds()
        image_file_names = [
            coco.loadImgs(image_id)[0]["file_name"] for image_id in image_ids
        ]

        # Create reference images
        output_path = os.path.join(cutouts_path, str(cat), "pixel_distributions")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        else:
            print(f"{output_path} already exists, not overwriting")

        for image_file_name in image_file_names:
            cutout_image_file_path = os.path.join(cutout_image_path, image_file_name)
            cutout_img_array = np.array(Image.open(cutout_image_file_path))
            pixel_dist = fill_image_with_pixel_dist(cutout_img_array)

            Image.fromarray(pixel_dist).save(os.path.join(output_path, image_file_name))


def transform_pixel_distributions(cutouts_path):
    """
    Transform the pixel distributions.
    :param cutouts_path: Path to the cutout images.
    :return: None.
    """
    for cat in range(NUM_CATEGORIES):

        train_annos = os.path.join(cutouts_path, str(cat), "annotations", "train.json")
        cutout_image_path = os.path.join(cutouts_path, str(cat), "train")
        output_path = os.path.join(cutouts_path, str(cat), "transformed_galaxy_cutouts")
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        else:
            print(f"{output_path} already exists, not overwriting")
        pixel_dists = os.path.join(cutouts_path, str(cat), "pixel_distributions")

        coco = COCO(train_annos)
        image_ids = coco.getImgIds()
        image_file_names = [
            coco.loadImgs(image_id)[0]["file_name"] for image_id in image_ids
        ]

        # Transform reference images to create radio galaxy cutouts with transformed pixel distributions.
        for image_file_name in image_file_names:

            img_array = np.array(Image.open(os.path.join(pixel_dists, image_file_name)))
            ref_image_file_name = random.choice(image_file_names)
            ref_image_path = os.path.join(pixel_dists, ref_image_file_name)

            # Alternatively, use the PixelDistributionAdaptation transform.
            # Can change the blend_ratio here , e.g. (0.1, 0.25)
            # transform = A.PixelDistributionAdaptation(
            #    [ref_image_path], blend_ratio=(0.25, 0.75), p=1
            # )

            transform = A.FDA([ref_image_path], beta_limit=(0.15, 0.3), p=1)

            transformed = transform(image=img_array)
            transformed_image = transformed["image"]

            cutout_img_array = np.array(
                Image.open(os.path.join(cutout_image_path, image_file_name))
            )
            for channel in range(img_array.shape[2]):
                transformed_image[:, :, channel][
                    np.where(cutout_img_array[:, :, channel] == BLACK_PIXEL)
                ] = 0

            output_image_file_path = os.path.join(output_path, image_file_name)

            Image.fromarray(transformed_image).save(output_image_file_path)


def fill_image_with_pixel_dist(imarr):
    new_imarr = deepcopy(imarr)
    for channel in range(imarr.shape[2]):
        # Check if all pixels are non-black
        if (
            imarr[:, :, channel][np.where(imarr[:, :, channel] != BLACK_PIXEL)].shape[0]
            > 0
        ):
            samples = sample_channel(
                imarr[:, :, channel][np.where(imarr[:, :, channel] != BLACK_PIXEL)]
            )
            new_imarr[:, :, channel][
                np.where(new_imarr[:, :, channel] == BLACK_PIXEL)
            ] = samples

    return new_imarr


def sample_channel(imarr):
    # imarr: One channel, IMAGE_HEIGHT by IMAGE_WIDTH
    return random.choices(imarr, k=(IMAGE_HEIGHT * IMAGE_WIDTH - imarr.shape[0]))
