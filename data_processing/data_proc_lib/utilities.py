import json
import os
import shutil

from ultralytics.data.converter import convert_coco

IMAGE_HEIGHT = 450
IMAGE_WIDTH = IMAGE_HEIGHT
BLACK_PIXEL = 0
NUM_CATEGORIES = 4


def create_directories(path, split_by_class=False, val_and_test_dirs=False):
    """
    Creates directories in the required structure, i.e.

    path
    |
    |--- annotations
    |--- train
    |--- val
    |--- test

    :param path: path in which to create the directories
    :split_by_class:
    :val_and_test_dirs:
    :return: None
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        print(f"{path} already exists, not overwriting")

    if split_by_class is True:
        if not os.path.isdir(os.path.join(path, "0", "train")):
            for i in range(NUM_CATEGORIES):
                os.makedirs(os.path.join(path, str(i), "train"))
                os.makedirs(os.path.join(path, str(i), "annotations"))
        else:
            print(f"Class directory 0 already exists, not overwriting")
    else:
        if not os.path.isdir(os.path.join(path, "annotations")):
            os.makedirs(os.path.join(path, "annotations"))
        else:
            print(f"Directory already exists, not overwriting")

        if not os.path.isdir(os.path.join(path, "train")):
            os.makedirs(os.path.join(path, "train"))
        else:
            print(f"train already exists, not overwriting")

        if val_and_test_dirs:
            if not os.path.isdir(os.path.join(path, "val")):
                os.makedirs(os.path.join(path, "val"))
            else:
                print(f"val already exists, not overwriting")

            if not os.path.isdir(os.path.join(path, "test")):
                os.makedirs(os.path.join(path, "test"))
            else:
                print(f"test already exists, not overwriting")


def combine(augmented_path, cleaned_path):
    """
    Combine an augmented dataset with the original cleaned images.
    :param augmented_path: Path to the augmented data.
    :param cleaned_path: Path to the cleaned data.
    :return:
    """
    combined_path = os.path.join(augmented_path, "combined")

    # Copy image and annotation files into combined directory
    shutil.copytree(
        os.path.join(augmented_path, "train"), os.path.join(combined_path, "train")
    )
    shutil.copytree(
        os.path.join(cleaned_path, "val"), os.path.join(combined_path, "val")
    )
    shutil.copytree(
        os.path.join(cleaned_path, "test"), os.path.join(combined_path, "test")
    )
    shutil.copytree(
        os.path.join(cleaned_path, "train"),
        os.path.join(combined_path, "train"),
        dirs_exist_ok=True,
    )
    os.mkdir(os.path.join(combined_path, "annotations"))
    shutil.copy(
        os.path.join(cleaned_path, "annotations", "val.json"),
        os.path.join(combined_path, "annotations"),
    )
    shutil.copy(
        os.path.join(cleaned_path, "annotations", "test.json"),
        os.path.join(combined_path, "annotations"),
    )

    # Combine the cleaned training annotations with the augmented annotations file
    with open(
        os.path.join(cleaned_path, "annotations", "train.json"), "r"
    ) as cleaned_file:
        data = cleaned_file.read()

    cleaned_train_data = json.loads(data)

    combined_train_data = {
        "info": cleaned_train_data["info"],
        "licenses": cleaned_train_data["licenses"],
        "images": [],
        "categories": cleaned_train_data["categories"],
        "annotations": [],
    }

    with open(
        os.path.join(augmented_path, "annotations", "train.json"), "r"
    ) as combined_file:
        data = combined_file.read()

    aug_train_data = json.loads(data)
    combined_train_data = cleaned_train_data
    combined_train_data["images"].extend(aug_train_data["images"])
    combined_train_data["annotations"].extend(aug_train_data["annotations"])

    with open(
        os.path.join(augmented_path, "combined", "annotations", "train.json"), "w"
    ) as f:
        json.dump(combined_train_data, f, ensure_ascii=False, indent=4)

    print(f"Completed! Combined data is in {os.path.join(augmented_path, 'combined')}.")


def convert_to_yolo(input_path, yolo_path):
    """
    Convert the COCO dataset to YOLO format.
    :param input_path: Path to the COCO format dataset.
    :param yolo_path: Path to place the YOLO formatted data.
    :return:
    """
    converted = os.path.join(input_path, "annotations", "converted")
    convert_coco(
        os.path.join(input_path, "annotations"),
        save_dir=converted,
        use_segments=True,
        cls91to80=False,
    )

    os.makedirs(os.path.join(yolo_path, "datasets", "images"))
    os.mkdir(os.path.join(yolo_path, "datasets", "labels"))
    shutil.copy("../../models/yolo/data.yaml", os.path.join(yolo_path, "datasets"))

    with open("../../models/yolo/data.yaml", "r") as template:
        with open(os.path.join(yolo_path, "datasets", "data.yaml"), "w") as newfile:
            newfile.write(f"path: {yolo_path}/datasets/\n")
            newfile.write(template.read())

    shutil.copytree(
        os.path.join(input_path, "train"),
        os.path.join(yolo_path, "datasets", "images", "train"),
    )
    shutil.copytree(
        os.path.join(input_path, "val"),
        os.path.join(yolo_path, "datasets", "images", "val"),
    )
    shutil.copytree(
        os.path.join(input_path, "test"),
        os.path.join(yolo_path, "datasets", "images", "test"),
    )

    shutil.copytree(
        os.path.join(converted, "labels", "train"),
        os.path.join(yolo_path, "datasets", "labels", "train"),
    )
    shutil.copytree(
        os.path.join(converted, "labels", "val"),
        os.path.join(yolo_path, "datasets", "labels", "val"),
    )
    shutil.copytree(
        os.path.join(converted, "labels", "test"),
        os.path.join(yolo_path, "datasets", "labels", "test"),
    )

    print(f"Completed! YOLO data is in {yolo_path}.")
