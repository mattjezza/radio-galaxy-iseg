# radio-galaxy-iseg

This repository contains code for augmenting radio galaxy image data for the task
of instance segmentation. The inspiration for this work
comes from [1]. This describes the RadioGalaxyNET dataset [2],
which is the source data used in developing this work.

1. `data_processing`: This section is a toolkit for augmenting data,
including creating synthetic galaxies. Library code is in directory `data_proc_lib`.
There is a README describing usage of the
library code and a `notebooks` directory containing Jupyter Notebooks that
walk a user through the process of using the library to augment data.
2. `models/yolo`: This contains notebooks concerned with training
and evaluating a YOLO model. There is a README describing usage. `yolo.ipynb`
is a Jupyter Notebook that walks through the process of training and evaluating
a model.

## References

1. Nikhel Gupta et al: "RadioGalaxyNET: Dataset and novel computer vision
   algorithms for the detection of extended radio galaxies and infrared hosts".
   In: Publications of the Astronomical Society of Australia 41 (2024).
2. N. Gupta, R. Norris, M. Huynh, Z. Hayder, and L. Petersson, 
“RadioGalaxyNET Dataset - Extended Radio Galaxies. v1. CSIRO.
Data Collection.” 2024. [Online]. Available: https://doi.org/10.25919/
btk3-vx79
