import cv2
import numpy as np
import os
from skimage import io
np.set_printoptions(suppress=True)
import math
import matplotlib.pyplot as plt # plt 用于显示图片
import time
import pandas as pd
from multiprocessing import Pool


value_length=5     ##取值的长度
value_step=1           ###取值的步长
search_step=1      ###搜寻的步长
f=0.92;f1=0.08                       ####   f1越大，边缘越多；f越大，边缘越少

starttime=time.time()



def FillHole(mask):                         #########白色为准
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out

####寻找最大连通区域#####
def find_max_region(mask_sel):
    contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 找到最大区域并填充
    area = []

    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))

    max_idx = np.argmax(area)

    max_area = cv2.contourArea(contours[max_idx])

    for k in range(len(contours)):

        if k != max_idx:
            cv2.fillPoly(mask_sel, [contours[k]], 0)
    return mask_sel

def memSim(img, f, f1):     #####mem函数第55---->66位灰度发生变化，其变化体现在mem函数的第67位发生突变
    last_num = np.repeat(img[:, -1], 1).reshape((-1, 1))
    img = np.concatenate((img[:, 0:], last_num), axis=1)

    l1 = np.expand_dims(np.array(img[:, 0]), axis=1)  # 得到图像的第一列
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

####得到图像的胸腔灰度的自适应阈值
def get_mask(binary,initial_threshold):
    ######### 自适应阈值（胸腔） #########
    binary = find_max_region(binary)  ###二值图像的最大连通区域，即胸腔
    c = np.multiply(np.mat(binary), np.mat(img))
    d = np.sum(c)
    number = len(c.nonzero()[0])
    condition = d / number
    ########双阈值，第一个阈值较大，第二个阈值较小##########第一个阈值为强边缘，第二个阈值为弱边缘######
    if initial_threshold<50:
        gradient_threshold = (3.264 * condition / 255) / 50
    else:
        gradient_threshold = (3.264 * condition / 255) / 20
    return gradient_threshold

###由原始图像取横向的和竖向的两个方向的值，得到一个二维矩阵，矩阵的每一格元素分别为【灰度值，横向取值，纵向取值】
def get_object_matrix(img):
    for y in range(scale, img.shape[0]-scale,value_step):      #####行数
        for x in range(scale, img.shape[1]-scale,value_step):    #####列数
            horizontal_element=img[x,y-scale:y+scale+1]               ####切片左闭右开
            new_matrix_h[x, y] = horizontal_element
            vertical_element=img[x-scale:x+scale+1,y]
            new_matrix_v[x, y] = vertical_element                   ######生成了两个存有横方向和纵方向上的value长度的信息
    return new_matrix_h,new_matrix_v

#######将矩阵的形式装换成竖着的可以直接输进mem函数的形式，即[m*n,7],新矩阵命名为new_matrix_mem
def trans_matrix(old_matrix):             ######    输入的矩阵为【512,512】，含有整数和矩阵对象
    # 将矩阵展平为一维形状,用以得到矩阵的长和宽的乘积
    flat_array = old_matrix.reshape(-1)
    # 将一维数组变形为原始形状中的列向量
    new_matrix = flat_array.reshape(flat_array.shape[0], -1)
    # 将每个列向量转成一个长度为？的行向量
    new_matrix_mem = np.zeros((new_matrix.shape[0], value_length))      ######
    shape=new_matrix.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            if isinstance(new_matrix[i][j], np.ndarray):
                new_matrix_mem[i] = new_matrix[i][j]
            else:
                new_matrix_mem[i] = np.zeros(value_length)
    return new_matrix_mem                                ##### 获得可直接输入memsim函数的矩阵


def get_gradient_matrix(piex_matrix,mem_matrix,threshold):         ##### 图像灰度矩阵，图像的取值矩阵，灰度阈值
    gradient_matrix=np.zeros_like(piex_matrix).astype(float)
    for i in range(piex_matrix.shape[0]):
        piex=piex_matrix[i,:]
        mem=mem_matrix[i,:]
        # 找到绝对值最大值
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
            if abs(dx)>=abs(dy) and dx>=0:        ####横向的最大值大于纵向的最大值，且横向的最大值大于0，即为向右的方向，此时方向为90度
                direction_matrix[i,j]=1
            elif abs(dx)>=abs(dy) and dx<0:
                direction_matrix[i, j] = 2
            if abs(dx)<abs(dy) and dx>=0:
                direction_matrix[i,j]=3
            elif abs(dx)<abs(dy) and dx<0:
                direction_matrix[i, j] = 4





    return gradient_matrix,direction_matrix




if __name__=="__main__":
    path_read = './pacienti_samples_squeezed/initial_image/'
    path_strong='./pacienti_samples_squeezed/gradient_matrix/'
    for filename in os.listdir(path_read):
        print('正在进行的图片名称：', filename)
        starttime1 = time.time()
        img = io.imread(path_read + '/' + filename,0)

        img_result = np.zeros_like(img)
        m = img.shape[0];n = img.shape[1]
        img = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
        new_matrix_h = np.zeros((m, n), dtype=object)
        new_matrix_v = np.zeros((m, n), dtype=object)
        scale = int((value_length - 1) / 2)

        initial_threshold, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  #####获得二值图像和阈值
        gradent_threshold = get_mask(binary,initial_threshold)        #####获得用于判断梯度阈值的参数
        initial_threshold = initial_threshold

        ####获得用于处理
        dst = FillHole(binary)  ####空洞填充得到整个胸腔
        kernel2 = np.ones((30, 30), np.uint8)
        Max_connected_area_fill = cv2.erode(dst, kernel2)  ###缩小胸腔，去除胸腔外的部分
        Max_connected_area_fill = Max_connected_area_fill / 255  ####处于255，用于和边缘相乘，去除内部边缘

        new_matrix_h, new_matrix_v = get_object_matrix(img)
        new_matrix_h_expend = trans_matrix(new_matrix_h)  #####横向取值
        new_matrix_v_expend = trans_matrix(new_matrix_v)  #####纵向取值
        pa_h = memSim(new_matrix_h_expend, f, f1)
        pa_h = np.delete(pa_h, 0, axis=1)  ######删除第一个列多余的数据
        pa_v = memSim(new_matrix_v_expend, f, f1)
        pa_v = np.delete(pa_v, 0, axis=1)  #######第二个对应的是第二个和第一个进行差分，列数为5，分别对应五个差分
        #####将原始矩阵先按行拼接，再调整成一列
        # 将矩阵展平为一维形状,用以得到矩阵的长和宽的乘积
        flat_array = img.reshape(-1)
        # 将一维数组变形为原始形状中的列向量
        piex_matrix = flat_array.reshape(flat_array.shape[0], -1)
        gradient_matrix_h = get_gradient_matrix(piex_matrix, pa_h, initial_threshold)
        gradient_matrix_v = get_gradient_matrix(piex_matrix, pa_v, initial_threshold)
        gradient_matrix_h = gradient_matrix_h.reshape(1, -1)
        gradient_matrix_v = gradient_matrix_v.reshape(1, -1)
        ####此矩阵为梯度矩阵
        gradient_matrix_h = gradient_matrix_h.reshape(img.shape[0], img.shape[1])
        gradient_matrix_v = gradient_matrix_v.reshape(img.shape[0], img.shape[1])

        gradient_matrix, direction_matrix = gradient_maxtrix(gradient_matrix_h, gradient_matrix_v)

        immmmmm = (abs(gradient_matrix) / np.max(abs(gradient_matrix)) * 255).astype(np.uint8)
        image_result = immmmmm * Max_connected_area_fill  ####去除胸腔
        cv2.imwrite(path_strong + filename, image_result)

        endtime=time.time()
        print(endtime-starttime1)
    print('2112个图片，共耗时：',endtime-starttime)
