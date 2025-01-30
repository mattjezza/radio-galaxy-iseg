# YOLO

The code in this section handles training and testing a YOLO instance segmentation model.

## Installation

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

`pip3 install pandas`

`pip3 install pycocotools`

`pip3 install supervision`

`pip3 install opencv-python`

Do not install Albumentations. This is because the Albumentations
package will automatically apply additional online augmentations
during training.

## Usage

See the notebook `yolo.ipynb` for details of how to train and evaluate the models.

# COCO Performance Metrics
Note that for comparison with the other models, we can't use YOLO's built-in
evaluation metrics. We have to evaluate with the COCO API. A function `coco_evaluate`
has been written to do this and is in `yolo.ipynb`. Note this is based on
`YOLO_RadioGalaxyNET/coco_evaluate.py` in https://github.com/Nikhel1/Gal-YOLOv8.
Several changes have been made to the original. We are grateful to the authors
of [1] for making this available under the MIT license, a copy of which is included
below.

# References

1. Nikhel Gupta et al: "RadioGalaxyNET: Dataset and novel computer vision
   algorithms for the detection of extended radio galaxies and infrared hosts".
   In: Publications of the Astronomical Society of Australia 41 (2024).

------
The following license and copyright applies to the coco_evaluate() function:
### MIT License

Copyright (c) 2023 Nikhel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
