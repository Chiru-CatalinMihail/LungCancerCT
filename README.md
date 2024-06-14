# LungCancerCT
My disertation project tackling Lung Cancer Segmentation in CT slices

# TODO DOCUMENTAT toate fisierele, ce fac, si cum sunt structurate fisierele


training_indices_stack=6_small.pkl -> It only takes frames in the interval [tumour_volume - 3 , min(volume_size, tumor_volume + 3)]

training_indices_stack=6_small.pkl -> Overlapping window in interval [0, volume_size]


TODO: Destelenit structura fisierelor .py, nu e ok sa existe dependente circulare intre ele!!