# LungCancerCT
My disertation project tackling Lung Cancer Segmentation in CT slices

# TODO DOCUMENTAT toate fisierele, ce fac, si cum sunt structurate fisierele


training_indices_stack=6_small.pkl -> It only takes frames in the interval [tumour_volume - 3 , min(volume_size, tumor_volume + 3)]

training_indices_stack=6_small.pkl -> Overlapping window in interval [0, volume_size]


TODO: Destelenit structura fisierelor .py, nu e ok sa existe dependente circulare intre ele!!


I use the terms "stacks" and "slices" interchangeably.


ENVIRONMENT SETUP:

PYTHON: 3.9.13

PYTORCH: 2.1.0+cu121

NUMPY: 1.22.3

OPENCV/CV2: 4.5.5

MONAI: 1.4.dev2418

ETC