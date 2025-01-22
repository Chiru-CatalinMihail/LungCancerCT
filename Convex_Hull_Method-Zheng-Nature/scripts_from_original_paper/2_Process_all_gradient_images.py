import numpy as np
import cv2
# import SegmentationMetric
import os
from skimage import io
np.set_printoptions(suppress=True)
########## 目的：去除图像中的血管等物体



def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

path_read = './pacienti_samples_squeezed/gradient_matrix/'
path_gray = './pacienti_samples_squeezed/prediction/'
for filename in os.listdir(path_read):
    print('正在进行的图片名称：', filename)
    img = io.imread(path_read + '/' + filename, 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    binary = FillHole(binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 计算平均面积
    areas = list()
    for i in range(num_labels):
        if stats[i][-1] > 2:
            areas.append(stats[i][-1])
    cccccc = areas
    cccccc.sort(reverse=True)

    # 筛选超过平均面积的连通域
    image_filtered = np.zeros_like(img)
    # 筛选位于指定区域外的连通域
    for (i, label) in enumerate(np.unique(labels)):
        if cccccc[1] < 500:
            if label == 0 or stats[i][-1] < 250:
                continue
        # 如果是背景或面积小于50个像素，忽略
        if label == 0 or stats[i][-1] < 50:
            continue
        centroid_x = centroids[i][0]
        centroid_y = centroids[i][1]

        if cccccc[1] < 4000:
            if 270 <= centroid_x <= 460 and ((80 <= centroid_y <= 256)) and stats[i][-1] < 1500:  ####去左上方的和右下方的
                continue
            if 323 <= centroid_x <= 400 and ((260 <= centroid_y <= 295)) and stats[i][-1] < 1700:  ####去左上方的和右下方的
                continue
        else:  ###此时肺部图形较为完善，可以采用较大的
            if 180 <= centroid_x <= 350 and 100 <= centroid_y <= 400 and stats[i][-1] < 400:  ####去除中心血管等

                continue
            if 200 <= centroid_x <= 330 and 190 <= centroid_y <= 305 and stats[i][-1] < 1100:  ####去除中心血管等
                continue  #######   中心的血管分为两部分，图像中心靠两边的较小，中心的较大

        image_filtered[labels == i] = 255

    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image_filtered = cv2.morphologyEx(image_filtered, cv2.MORPH_CLOSE, kernel1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_filtered, connectivity=8)

    # 计算平均面积
    areas = list()
    for i in range(num_labels):
        if stats[i][-1] > 2:
            areas.append(stats[i][-1])
    cccccc = areas
    cccccc.sort(reverse=True)
    image_filtered_result = np.zeros_like(image_filtered)
    # 筛选位于指定区域外的连通域
    for (i, label) in enumerate(np.unique(labels)):
        if label == 0 or stats[i][-1] < 60:
            continue
        centroid_x = centroids[i][0]
        centroid_y = centroids[i][1]
        if cccccc[1] >= 4000:  ###此时肺部图形较为完善，可以采用较大的
            if 270 <= centroid_x <= 420 and ((80 <= centroid_y <= 256)) and stats[i][-1] < 2000:  ####去右上方的和右下方的
                continue
            if 350 <= centroid_x <= 370 and ((140 <= centroid_y <= 165)) and stats[i][-1] < 2500:  ####右下方的
                continue
            if 200 <= centroid_x <= 340 and ((400 <= centroid_y <= 480) or (30 <= centroid_y <= 120)) and stats[i][
                -1] < 1500:  ####去除两肺间的间层
                continue
            if ((10 <= centroid_x <= 130) or (392 <= centroid_x <= 510)) and 0 <= centroid_y <= 511 and stats[i][-1] < 800:  ####去除两侧的
                continue
            if 392 <= centroid_x <= 510 and 230 <= centroid_y <= 272 and stats[i][-1] < 2000:  ####去除两侧的
                continue
            if 330 <= centroid_x <= 390 and 210 <= centroid_y <= 280 and stats[i][-1] < 1500:  ####去除中心靠右位于肺实质内部的
                continue
            if 200 <= centroid_x <= 300 and 330 <= centroid_y <= 380 and stats[i][-1] < 1000:  ####去除中心靠底部的
                continue

        image_filtered_result[labels == i] = 255
    image_filtered = FillHole(image_filtered)
    cv2.imwrite(path_gray + filename, image_filtered_result)




