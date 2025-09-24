import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion, generate_binary_structure
import torch
import os
import glob

plt.style.use('ggplot')
matplotlib.use( 'tkagg' )


# TODO Optimize here - parallelize the code
def calculate_local_fractal_dimensions(matrix, box_size=5, step=2):
    """
    Aplică metoda box-counting optimizată* / max pe fiecare pixel dintr-o subsecțiune a matricei.
    :param matrix: Matrice 2D care conține o imagine.
    :param box_size: Dimensiunea ferestrei în jurul fiecărui pixel pentru calculul fractal.
    :param step: Pasul pentru a reduce granularitatea (ex. procesare la fiecare al 2-lea pixel).
    :return: Matrice 2D cu dimensiunile fractale locale calculate parțial.
    """
    height, width = matrix.shape
    result_matrix = np.zeros((height, width))
    result_matrix_max = np.zeros((height, width))

    # Precalculăm logaritmii pentru dimensiuni
    sizes = np.arange(1, box_size + 1)

    print(f'Sizes: {sizes}')

    log_sizes = -np.log(sizes)

    for row in range(0, height, step):
        for col in range(0, width, step):
            # Definim fereastra locală
            start_row = max(row - box_size // 2, 0)
            end_row = min(row + box_size // 2 + 1, height)
            start_col = max(col - box_size // 2, 0)
            end_col = min(col + box_size // 2 + 1, width)

            #TODO: Asta merge scoasa in afara si calculata o singura data pentru toata iimaginea pentru a  optimiza si mai mult.
            region = matrix[start_row:end_row, start_col:end_col]

            # Normalizează valorile la 0 și 1
            binary_region = (region > 0).astype(int)

            # Metoda box-counting
            box_counts, box_counts_max = [], []

            for size in sizes:
                count = 0
                count_max = 0
                max_value = -1
                for r in range(0, binary_region.shape[0], size):
                    for c in range(0, binary_region.shape[1], size):
                        sub_box = binary_region[r:r + size, c:c + size]
                        if np.sum(sub_box) > 0:
                            count += 1

                        if (size > 1):
                            max_value = np.max(sub_box)
                            # Împarte valoarea maximă la dimensiunea cutiei
                            quotient = max_value // box_size
                            remainder = max_value % box_size
                            count_max += quotient
                            if remainder != 0:
                                count_max += 1
                        else:
                        # Pentru cutii de dimensiunea 1x1, doar adaugă valoarea
                            count_max += matrix[row][col]

                box_counts.append(count)
                box_counts_max.append(count_max)

            # Verificăm dacă box_counts conține valori zero
            if len(box_counts) < 2 or np.any(np.array(box_counts) == 0):
                result_matrix[row, col] = 0  # Evită calculul logaritmilor pe date invalide
                continue

            if len(box_counts_max) < 2 or np.any(np.array(box_counts_max) == 0):
                result_matrix_max[row, col] = 0
                continue

            log_counts = np.log(np.maximum(box_counts, 1))
            log_counts_max = np.log(np.maximum(box_counts_max, 1))

            # Verificăm dacă log_sizes și log_counts conțin valori valide
            if np.any(np.isnan(log_counts)) or np.any(np.isinf(log_counts)) or np.any(log_counts == 0):
                result_matrix[row, col] = 0
                continue

            if np.any(np.isnan(log_counts_max)) or np.any(np.isinf(log_counts_max)) or np.any(log_counts_max == 0):
                result_matrix_max[row, col] = 0
                continue

            # Calcul regresie liniară pe punctele log-log
            slope, _ = np.polyfit(log_sizes, log_counts, 1)
            result_matrix[row, col] = slope

            slope_max, _ = np.polyfit(log_sizes, log_counts_max, 1)
            result_matrix_max[row, col] = slope_max


    fractal_map = torch.cat((torch.tensor(result_matrix, dtype=torch.float32).unsqueeze(0), torch.tensor(result_matrix_max, dtype=torch.float32).unsqueeze(0)), dim=0)

    print(f'Fractal map shape: {fractal_map.shape}')

    return fractal_map


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


def unnormalize_matrix(matrix, min_value, max_value):
    """
    Reia matricea normalizată la intervalul original.
    :param matrix: Matricea normalizată.
    :param min_value: Valoarea minimă din intervalul original.
    :param max_value: Valoarea maximă din intervalul original.
    :return: Matricea reescalată.
    """
    return matrix * (max_value - min_value) + min_value


# TODO Optimize here - Multiprocessing for each patient

if __name__ == '__main__':
    crt_folder = './initial_image'
    files = glob.glob(crt_folder + '/*.pt')

    for file_path in files:

        print(f'Processing file: {file_path}')

        data = torch.load(file_path)

        print(f'Data shape: {data.shape}')
        # print(file_path[:-3])

        fractal_maps = torch.zeros((2, data.shape[1], data.shape[2], data.shape[3]))

        for slice_idx in range(data.shape[-1]):
            print(f'Slice {slice_idx + 1} / {data.shape[-1]}')

            slice_data = data[0, :, :, slice_idx].squeeze().numpy()
            print(f'Slice {slice_idx} min: {slice_data.min()}, max: {slice_data.max()}')

            # Want to keep the original image shape [CHANNELS = 1, HEIGHT = 512, WIDTH = 512, SLICES = VARIABLE]

            raw_image = unnormalize_matrix(slice_data, min_value=-1024, max_value=1462)
            
            print(f'Raw image min: {raw_image.min()}, max: {raw_image.max()}')

            # harta fractală
            fractal_map = calculate_local_fractal_dimensions(raw_image, box_size=3, step=1) # Andreea suggested box_size = 8, return it to that after parallelizing the code

            fractal_maps[:, :, :, slice_idx] = fractal_map
                        

            fig, axes = plt.subplots(1, 3, figsize=(15, 10))

            for ax in axes:
                ax.axis('off')
                ax.imshow(raw_image, cmap='gray')

            axes[0].imshow(slice_data, cmap='viridis', alpha=0.5)
            axes[1].imshow(fractal_map[0, :, :], cmap='viridis', alpha=0.5)
            axes[2].imshow(fractal_map[1, :, :], cmap='viridis', alpha=0.5)

            plt.savefig(f"{file_path[:-3]}_slice_{slice_idx}_fractal_map.png")
                

        torch.save(fractal_maps, f"{file_path[:-3]}_fractal_maps.pt")




