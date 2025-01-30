import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion, generate_binary_structure
import torch
import os
from skimage.filters import threshold_otsu
import glob

def extract_contour(matrix, threshold):
    """
    Extrage conturul unei leziuni dintr-o matrice pe baza unui prag dat.
    :param matrix: Matricea 2D care conține datele imaginii.
    :param threshold: Pragul pentru a detecta regiunea leziunii.
    :return: Matrice binară care conține doar conturul leziunii.
    """
    # Creează o mască binară pe baza pragului
    binary_mask = matrix > threshold

    # Structura pentru eroziune (vecinătatea)
    structure = generate_binary_structure(2, 1)

    # Eroziunea binară pentru a scoate interiorul și păstra doar conturul
    eroded_mask = binary_erosion(binary_mask, structure=structure)

    # Contur = diferența între mască și eroziunea ei
    contour = binary_mask.astype(int) - eroded_mask.astype(int)
    return contour



#file_path = 'patient0_slice340_mare'
#file_path = 'patient0_slice341_mare'
#file_path = 'patient0_slice342_mare'
#file_path = 'patient0_slice343_mare'
#file_path = 'patient1_slice180_mic'
#file_path = 'patient1_slice215_mic'

files = glob.glob('pacienti_cu_gradcam/*_with_fractal_map.pt')


for file in files:
    data = torch.load(file)
    print(data.shape, torch.max(data[0]), torch.min(data[0]))

    first_matrix = data[0].numpy().squeeze(-1)
    second_matrix = data[1].numpy().squeeze(-1)
    fractal_channel1 = data[4].numpy().squeeze(-1)
    fractal_channel2 = data[5].numpy().squeeze(-1)

    #  contur leziune
    threshold = threshold_otsu(second_matrix)
    contour = extract_contour(second_matrix, threshold)

    fractal_map_norm = Normalize(vmin=fractal_channel1.min(), vmax=fractal_channel1.max())
    fractal_map_max_norm = Normalize(vmin=fractal_channel2.min(), vmax=fractal_channel2.max())


    titles = [['Raw Image', 'Ground Truth'], ['GradCAM', 'GradCAM thresholded'], ['Fractal Map', 'Fractal Map Max']]

    images = []

    i = 0

    for _ in range(3):
        crt_row = []
        for _ in range(2):
            crt_row.append(data[i].numpy().squeeze(-1))
            i += 1
        images.append(crt_row)


    fig, axs = plt.subplots(len(titles), 2, figsize=(15, 15))

    for i in range(len(titles)):
        for j in range(2):
            axs[i, j].axis('off')
            axs[i, j].imshow(images[i][j])
            axs[i, j].set_title(titles[i][j])
            axs[i, j].set_aspect('auto')

    plt.savefig('./' + file[:-3].split('/')[-1] + '.png')
    plt.close()