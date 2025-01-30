# Data Preprocessing and Augmentation

The code in this section handles data preprocessing (such as data cleaning) and
the creation of offline data augmentations.

## 1 Installation

* Step 1: Create a Python 12 venv:
  (Note, ensure Python 12 is installed on the system as a prerequisite.)

`python3.12 -m venv ./venv`

Activate the environment.

`source ./venv/bin/activate`

* Step 2: Install the packages for Pytorch according to the [instructions here](https://pytorch.org/).

For example, to install the latest PyTorch on a cuda 11.8 platform, use:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

* Step 3: Install some useful packages:

`pip3 install ultralytics`

`pip3 install -U albumentations`

`pip3 install pandas`

`pip3 install pycocotools`

`pip3 install supervision`

`pip3 install opencv-python`

Note this environment is not suitable for running YOLO. This is because the Albumentations
module installed in this environment will automatically apply additional online augmentations
during training. We found this to be detrimental to the final model performance.

## 2 Using the Tools to Create Offline Augmented Datasets

### 2.1 Download the Data

Download the data from [this link](https://doi.org/10.25919/btk3-vx79).
Unzip the file. The images and annotations will be in a directory called `data`.
The Jupyter notebooks assume the `data` directory is in the same directory
as this repo, i.e. directory `data` and `radio_galaxy_iseg` are sibling directories
in the same parent directory.

### 2.2 Clean the Data

This is an essential prerequisite for subsequent steps!

See the Jupyter notebook `notebooks/clean.ipynb`.

Assuming you keep the same value for the `CLEANED_PATH` in this script then the
output will be placed in a directory named `cleaned`. This will be a sibling
directory to `data` and `RadioGalaxyISeg`.

You can train a model with the cleaned dataset (i.e. without further augmentations).
Mask R-CNN and CenterMask2 can use the dataset as it is. YOLO requires an extra step -
we need to convert the COCO data format files into YOLO format. Instructions for
this are included in `clean.ipynb`.

### 2.3 Create a Transformed Dataset

This step takes the cleaned dataset and applies transformations to augment it.

See the Jupyter notebook `notebooks/rotate.ipynb`.

This notebook also includes instructions to combine the augmented dataset with
the cleaned data and to convert to YOLO format.

At this point, the augmented dataset is ready to use in model training.

## 3 Creating Synthetic Galaxies and Adding them to Training Images

Prerequisites: the data needs to be downloaded and cleaned. See the sections above to do that.

There are two steps to create synthetic galaxy images.

1. Create synthetic galaxies
2. Paste synthetic galaxies into existing images.

### 3.1 Create Synthetic Galaxies

Run `notebooks/create_synthetic_galaxies.ipynb`

This requires the user to provide two paths:

* `CLEANED_PATH`: The path to the output from the `notebooks/clean.ipynb`. If you
  kept the default value, the directory name will be `cleaned`, you just need to
  complete the path including `cleaned`.

* `CUTOUTS_PATH`: Full path to the directory where the cutouts and pixel transforms will be stored.

### 3.2 Paste Synthetic Galaxies into Training Images

Run `notebooks/paste_synthetic_galaxies.ipynb`, section 1. Use the *same* paths for
`CLEANED_PATH` and `CUTOUTS_PATH` as in steps 3.1 above.

After completing this you'll have an augmented dataset containing training images
with additional synthetic galaxies pasted into them.

You can repeat this process to add more than one synthetic galaxy to each
training image. There are instructions to do this in
`notebooks/paste_synthetic_galaxies.ipynb`

After this, you probably want to augment it further. The path we followed in the study is:

1. Combine the dataset with the synthetic galaxies pasted in with the original
   (cleaned) dataset
2. Next, further augment the combined dataset by rotation, cropping and scaling.
3. Combine the resulting dataset with the original cleaned dataset.

Instructions for all of these, and converting to YOLO format, are included in
`notebooks/paste_synthetic_galaxies.ipynb`.

### 3.3 Other Methods of Generating New Galaxies

Generative models (GAN, VAE) can be used to create new galaxies.

1. Create a training set of cutout images scaled to a constant size (we use 32x32).
2. Use these as input to the generative model of your choice. We have experimented with using a GAN and a VAE.

## 4 Troubleshooting

### 4.1 Setting Paths

When using the functions to create augmented datasets, you'll sometimes need to
set paths to input directories (where the source files are) and output directories
(where the output will go). This is clearly signposted and explained in the notebooks.
Default paths are set in the notebooks which may work unchanged, but due to
differences between individual environments this is not guaranteed. So,
you'll probably need to set some paths to match your own computer where the
code is run.

If something goes wrong during the process of creating the augmented data (for
any reason) then you may end up with an output directory that contains some of
the outputs you want but not all of them. In that case, if you want to try again,
the best course of action is either to delete the output directory completely
or change the output path to a new directory and re-run. That way, you know
you're recreating the data cleanly, which is more likely to work
smoothly.

Most paths in the notebooks can be relative or absolute. However, using the
`convert_to_yolo` function is an exception. The `YOLO_PATH` should be absolute.

### Registering COCO Instances with `register_coco_instances`

To use the COCO format dataset, we frequently use `register_coco_instances`.
This includes a name for the dataset. Note that a name can be registered only
once. Attempting to re-register the same name will result in an error.
Sometimes, if something goes wrong, or if you just want to change a value
and see its effect, you may want to re-run `register_coco_instances`.
If the same name is used then you'll see an error like "dataset <name> already
registered". You can solve this in a couple of ways. One way is to change the
dataset name in the call to `register_coco_instances`,
which will allow it to be re-run with the new name. Note, however, that any
subsequent code that relies on that dataset name will also need to be changed
to use the new name.

Another way to solve this problem without changing the dataset name is to simply
restart the kernel of the Jupyter Notebook. This will clear the dataset name and
allow `register_coco_instances` to be re-run successfully.








