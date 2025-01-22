import cv2
import numpy as np
import os
from skimage import io
np.set_printoptions(suppress=True)
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from multiprocessing import Pool
# import tensorflow
import torch

value_length=5     
value_step=1           ###value step size
search_step=1      
f=0.92;f1=0.08                       ####   The larger f1 is, the more edges there are; the larger f is, the fewer edges there are.

starttime=time.time()

def convert_tensor_to_rgb(tensor):
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = (tensor * 255).astype('uint8')
    return tensor


def convert_pt_volume_to_jpg_images(filename):
    torch_volume = torch.load(filename)

    print("Torch volume shape", torch_volume.shape)

    stack_size = torch_volume.shape[-1]
    print('Stack size:', stack_size)

    images = [convert_tensor_to_rgb(torch_volume[:, :, :, i].squeeze().cpu().numpy()) for i in range(stack_size)]
    return images

def convert_images_to_pt_masks_volume(images):
    images = [torch.tensor(x) for x in images]
    volume = torch.stack(images)
    return volume


def FillHole(mask):                         ######### White background
    mask = cv2.convertScaleAbs(mask) # Repairs cv2 assertion from CV_8UC1 to CV_32SC1
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

####Find the maximum connected area#####
def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Find the largest area and fill it
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

def memSim(img, f, f1):     #####The gray level of bit 55---->66 of the mem function changes, and the change is reflected in the mutation of bit 67 of the mem function.
    last_num = np.repeat(img[:, -1], 1).reshape((-1, 1))
    img = np.concatenate((img[:, 0:], last_num), axis=1)

    l1 = np.expand_dims(np.array(img[:, 0]), axis=1)  # Get the first column of the image
    l2 = np.expand_dims(np.array(img[:, 0]), axis=1)
    l3 = np.expand_dims(np.array(img[:, 0]), axis=1)

    fgr1 = f - f1
    fgr2 = f
    fgr3 = f + f1
    mem1 = 1 - fgr1
    mem2 = 1 - fgr2
    mem3 = 1 - fgr3
    k=img.shape[1]

    for i in range(1, k, 1):
        l1 = np.column_stack((l1, list(mem1 * l1[:, i - 1] + fgr1 * img[:, i])))
        l2 = np.column_stack((l2, list(mem2 * l2[:, i - 1] + fgr2 * img[:, i])))
        l3 = np.column_stack((l3, list(mem3 * l3[:, i - 1] + fgr3 * img[:, i])))

    res = 2 * l2 - l1 - l3
    return res

#### Get the gradient thresholded mask of the image
def get_mask(binary,initial_threshold):
    ######### Adaptive Threshold (Chest) #########
    binary = find_max_region(binary)  ###The largest connected region of the binary image, i.e. the pulmonar cavity
    c = np.multiply(np.mat(binary), np.mat(img))
    d = np.sum(c)
    number = len(c.nonzero()[0])
    condition = d / number
    # Double threshold, the first threshold is larger, the second threshold is smaller
    # The first threshold is a strong edge, the second threshold is a weak edge
    if initial_threshold<50:
        gradient_threshold = (3.264 * condition / 255) / 50
    else:
        gradient_threshold = (3.264 * condition / 255) / 20
    return gradient_threshold

### The horizontal and vertical values ​​of the original image are taken to obtain a two-dimensional matrix. Each element of the matrix is ​​[grayscale value, horizontal value, vertical value]
def get_object_matrix(img):
    for y in range(scale, img.shape[0]-scale,value_step):      ##### Rows
        for x in range(scale, img.shape[1]-scale,value_step):    ##### Columns
            horizontal_element=img[x,y-scale:y+scale+1]               #### [left, right)
            new_matrix_h[x, y] = horizontal_element
            vertical_element=img[x-scale:x+scale+1,y]
            new_matrix_v[x, y] = vertical_element                   ###### Generates two values ​​with the length of the value in the horizontal and vertical directions

    return new_matrix_h,new_matrix_v

#######Convert the matrix into a vertical form that can be directly input into the mem function, that is [m*n,7] = new_matrix_mem
def trans_matrix(old_matrix):             ######    Input shape = [512,512]
    # Flatten the matrix into a one-dimensional shape to get the product of the length and width of the matrix
    flat_array = old_matrix.reshape(-1)
    # 将一维数组变形为原始形状中的列向量
    new_matrix = flat_array.reshape(flat_array.shape[0], -1)
    # Reshape a 1D array into a column vector in the original shape
    new_matrix_mem = np.zeros((new_matrix.shape[0], value_length)) 

    shape=new_matrix.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if isinstance(new_matrix[i][j], np.ndarray):
                new_matrix_mem[i] = new_matrix[i][j]
            else:
                new_matrix_mem[i] = np.zeros(value_length)
    return new_matrix_mem                                ##### Get a matrix that can be directly input into the memsim function


def get_gradient_matrix(piex_matrix,mem_matrix,threshold):         ##### Params: piex_matrix: The original image matrix, mem_matrix: The matrix after mem function, threshold: The threshold of the original image
    gradient_matrix=np.zeros_like(piex_matrix).astype(float)
    for i in range(piex_matrix.shape[0]):
        piex=piex_matrix[i,:]
        mem=mem_matrix[i,:]
        # Find the absolute maximum
        max_val = np.max(mem)
        min_val = np.min(mem)
        if abs(max_val)>=abs(min_val):
            max_point_value=max_val
        else:
            max_point_value=min_val
        if piex<=threshold and abs(max_point_value)>= gradent_threshold:

            gradient_matrix[i,0]=max_point_value
    return gradient_matrix

def gradient_maxtrix(gradient_matrix_h,gradient_matrix_v):
    gradient_matrix=np.zeros_like(gradient_matrix_h)
    direction_matrix=np.zeros_like(gradient_matrix)
    for i in range(gradient_matrix_h.shape[0]):
        for j in range(gradient_matrix_v.shape[1]):
            dx=gradient_matrix_h[i,j]
            dy=gradient_matrix_v[i,j]
            gradient_matrix[i,j]=np.sqrt(dx ** 2 + dy ** 2)
            if dx==float(0) and dy==0.0:
                direction_matrix[i, j] = 0
            if abs(dx)>=abs(dy) and dx>=0:        ####The maximum value of the horizontal direction is greater than the maximum value of the vertical direction, and the maximum value of the horizontal direction is greater than 0, that is, the direction to the right, at this time the direction is 90 degrees
                direction_matrix[i,j]=1
            elif abs(dx)>=abs(dy) and dx<0:
                direction_matrix[i, j] = 2
            if abs(dx)<abs(dy) and dx>=0:
                direction_matrix[i,j]=3
            elif abs(dx)<abs(dy) and dx<0:
                direction_matrix[i, j] = 4

    return gradient_matrix,direction_matrix


#### Step 2 - Processing all gradient images

# import SegmentationMetric
########## Purpose: Remove objects such as blood vessels from images


##### Step 3 - Convex hull repair contour
def remove_longest_segments(contours):
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area >= 5000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:
                    if (length > 100 or length > ccccccccccccccc[1]):           ####Remove the longest one
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)
        if area <=2000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:
                    if length > ccccccccccccccc[2] :  #####Remove the two longest ones
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)


        if area > 2000 and area <5000:
            hull_points = cv2.convexHull(contours[i], returnPoints=True)
            lengths = [((hull_points[j + 1][0][0] - hull_points[j][0][0]) ** 2 +
                        (hull_points[j + 1][0][1] - hull_points[j][0][1]) ** 2) ** 0.5 for j in
                       range(len(hull_points) - 1)]
            ccccccccccccccc = lengths

            ccccccccccccccc.sort(reverse=True)
            idx = sorted(range(len(lengths)), key=lambda x: lengths[x], reverse=True)[:2]
            for j in range(len(hull_points) - 1):
                start_point = tuple(hull_points[j][0])
                end_point = tuple(hull_points[j + 1][0])
                length = ((end_point[0] - start_point[0]) ** 2 +
                          (end_point[1] - start_point[1]) ** 2) ** 0.5
                if len(lengths) >= 2:

                    if (length > 70 or length > ccccccccccccccc[2]):  #####Remove the two longest ones
                        continue
                    else:
                        cv2.line(img1, start_point, end_point, (255, 255, 255), thickness=1)

#########################    MAIN    #########################

if __name__=="__main__":
    path_read = './pacienti_samples_squeezed/initial_image/'
    path_strong='./pacienti_samples_squeezed/prediction_compact/'
    for filename in os.listdir(path_read):

        print('File analysed：', filename)
        images = convert_pt_volume_to_jpg_images(path_read + filename)

        masks = []

        for idx, img in enumerate(images):
            print(f'Processing image {idx + 1}/{len(images)}')
            starttime1 = time.time()

            img_result = np.zeros_like(img)
            m = img.shape[0];n = img.shape[1]
            img = cv2.GaussianBlur(img, (5, 5), 1)
            new_matrix_h = np.zeros((m, n), dtype=object)
            new_matrix_v = np.zeros((m, n), dtype=object)
            scale = int((value_length - 1) / 2)

            initial_threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gradent_threshold = get_mask(binary, initial_threshold)        ##### Gradent Threshold is used implicitly as a parameter in get_gradient_matrix
            initial_threshold = initial_threshold

            dst = FillHole(binary)  ####Fill the entire chest cavity
            kernel2 = np.ones((30, 30), np.uint8)
            Max_connected_area_fill = cv2.erode(dst, kernel2)  ###Reduce the size of the chest cavity, remove the parts outside of it

            Max_connected_area_fill = Max_connected_area_fill / 255  #### Normalize to 255, used to multiply with the edge to remove the internal edge

            new_matrix_h, new_matrix_v = get_object_matrix(img)
            new_matrix_h_expend = trans_matrix(new_matrix_h)  #####Horizontal value
            new_matrix_v_expend = trans_matrix(new_matrix_v)  #####Vertical value
            pa_h = memSim(new_matrix_h_expend, f, f1)
            pa_h = np.delete(pa_h, 0, axis=1)  ###### Delete the redundant data in the first column
            pa_v = memSim(new_matrix_v_expend, f, f1)
            pa_v = np.delete(pa_v, 0, axis=1)  ####### Difference between the second and the first, the number of columns is 5, corresponding to five differences
            #####Concatenate the original matrix into rows and then adjust it into a column
            # Flatten the matrix into a one-dimensional shape to get the product of the length and width of the matrix
            flat_array = img.reshape(-1)

            # Reshape a 1D array into a column vector in the original shape
            piex_matrix = flat_array.reshape(flat_array.shape[0], -1)
            gradient_matrix_h = get_gradient_matrix(piex_matrix, pa_h, initial_threshold)
            gradient_matrix_v = get_gradient_matrix(piex_matrix, pa_v, initial_threshold)
            gradient_matrix_h = gradient_matrix_h.reshape(1, -1)
            gradient_matrix_v = gradient_matrix_v.reshape(1, -1)

            gradient_matrix_h = gradient_matrix_h.reshape(img.shape[0], img.shape[1])
            gradient_matrix_v = gradient_matrix_v.reshape(img.shape[0], img.shape[1])

            ####This matrix is ​​the gradient matrix
            gradient_matrix, direction_matrix = gradient_maxtrix(gradient_matrix_h, gradient_matrix_v)

            immmmmm = (abs(gradient_matrix) / np.max(abs(gradient_matrix)) * 255).astype(np.uint8)
            image_result = immmmmm * Max_connected_area_fill  ####Removal of the chest cavity


            ### Step 2 - Processing all gradient images
            img_step2 = image_result
            _, binary = cv2.threshold(img_step2, 0, 255, cv2.THRESH_BINARY)
            binary = FillHole(binary)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

            # Calculate the average area
            areas = list()
            for i in range(num_labels):
                if stats[i][-1] > 2:
                    areas.append(stats[i][-1])
            cccccc = areas
            cccccc.sort(reverse=True)

            # Screening for connected domains with an area exceeding the average
            image_filtered = np.zeros_like(img_step2)

            # Filter connected domains outside the specified area
            for (i, label) in enumerate(np.unique(labels)):
                if cccccc[1] < 500:
                    if label == 0 or stats[i][-1] < 250:
                        continue
                # If it is background or the area is less than 50 pixels, ignore it.
                if label == 0 or stats[i][-1] < 50:
                    continue
                centroid_x = centroids[i][0]
                centroid_y = centroids[i][1]

                if cccccc[1] < 4000:
                    if 270 <= centroid_x <= 460 and ((80 <= centroid_y <= 256)) and stats[i][-1] < 1500:  ####Go to the upper left and lower right
                        continue
                    if 323 <= centroid_x <= 400 and ((260 <= centroid_y <= 295)) and stats[i][-1] < 1700:  ####Go to the upper left and lower right

                        continue
                else:  ###The lung image is more complete at this time, and a larger one can be used
                    if 180 <= centroid_x <= 350 and 100 <= centroid_y <= 400 and stats[i][-1] < 400:  ####Removal of central blood vessels, etc.
                        continue
                    if 200 <= centroid_x <= 330 and 190 <= centroid_y <= 305 and stats[i][-1] < 1100:  ####Removal of central blood vessels, etc.
                        continue  #######   The blood vessels in the center are divided into two parts. The ones on the sides of the center of the image are smaller, and the ones in the center are larger.

                image_filtered[labels == i] = 255

            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            image_filtered = cv2.morphologyEx(image_filtered, cv2.MORPH_CLOSE, kernel1)
            image_filtered = image_filtered.astype('uint8')

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_filtered) # Removed connectivity = 8 to repair OpenCV error (-215:Assertion failed) iDepth == CV_8U || iDepth == CV_8S



            # Calculate the average area
            areas = list()
            for i in range(num_labels):
                if stats[i][-1] > 2:
                    areas.append(stats[i][-1])
            cccccc = areas
            cccccc.sort(reverse=True)
            image_filtered_result = np.zeros_like(image_filtered)

            # Filter connected domains outside the specified area
            for (i, label) in enumerate(np.unique(labels)):
                if label == 0 or stats[i][-1] < 60:
                    continue
                centroid_x = centroids[i][0]
                centroid_y = centroids[i][1]
                if cccccc[1] >= 4000:  ###The lung image is more complete at this time, and a larger one can be used
                    if 270 <= centroid_x <= 420 and ((80 <= centroid_y <= 256)) and stats[i][-1] < 2000:  ####Go to the upper right and lower right
                        continue
                    if 350 <= centroid_x <= 370 and ((140 <= centroid_y <= 165)) and stats[i][-1] < 2500:  ####Bottom right
                        continue
                    if 200 <= centroid_x <= 340 and ((400 <= centroid_y <= 480) or (30 <= centroid_y <= 120)) and stats[i][
                        -1] < 1500:  ####去除两肺间的间层
                        continue
                    if ((10 <= centroid_x <= 130) or (392 <= centroid_x <= 510)) and 0 <= centroid_y <= 511 and stats[i][-1] < 800:  ####Remove the two sides
                        continue
                    if 392 <= centroid_x <= 510 and 230 <= centroid_y <= 272 and stats[i][-1] < 2000:  ####Remove the two sides
                        continue
                    if 330 <= centroid_x <= 390 and 210 <= centroid_y <= 280 and stats[i][-1] < 1500:  ####Remove the center right inside the lung parenchyma
                        continue
                    if 200 <= centroid_x <= 300 and 330 <= centroid_y <= 380 and stats[i][-1] < 1000:  ####Remove the center bottom
                        continue

                image_filtered_result[labels == i] = 255
            image_filtered = FillHole(image_filtered)

            if np.max(image_filtered_result)==0:
                image_filtered=image_filtered_result
            else:
                img1 = image_filtered_result.copy()
                # Calculate the number of non-zero connected domains
                nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img1, connectivity=8)

                areas = stats[1:, cv2.CC_STAT_AREA]  # Ignore background connected areas
                max_area_index = np.argmax(areas) + 1  # Get the index of the largest connected region excluding the background

                max_area = areas[max_area_index - 1]  # Get the area of ​​the largest connected domain
                print("Max area",  max_area)
                if nLabels <= 3 and max_area > 2000:  ####    The graph is large and there are some that need to be connected
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                if nLabels > 3 and max_area > 2000:  ####       The graph is large and needs to be connected more
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                if nLabels > 3 and max_area <= 2000:  ####      Smaller picture
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                if nLabels <= 3 and max_area <= 2000:  ####     Smaller images with better effects
                    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel1)
                # Find contours and compute convex hull
                contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                remove_longest_segments(contours)
                img1 = FillHole(img1)
                image_filtered=img1

                if type(img1)==int:
                    image_filtered=np.zeros_like(image_filtered_result)
                else:
                    pass

                # cv2.imwrite(path_strong + filename, image_result)
                
                endtime=time.time()
                print(endtime-starttime1)
            # Debug aici, dupa 2 imagini -> [1, 512, 512, 4] corect, [1, 512, 512, 3] != 4 missmatch verificat de ce
            masks.append(image_filtered)
            print('Spent time',endtime-starttime)

        torch_volume = convert_images_to_pt_masks_volume(masks).unsqueeze(0)
        torch_volume = torch_volume.permute(0, 2, 3, 1)

        print("Torch volume shape", torch_volume.shape)

        torch.save(torch_volume, path_strong + filename)