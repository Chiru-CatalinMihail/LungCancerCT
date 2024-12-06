






path_read = './pacienti_samples_squeezed/prediction/'
path_gray = './pacienti_samples_squeezed/prediction_convex/'


for filename in os.listdir(path_read):
    print('正在进行的图片名称：', filename)
    img = io.imread(path_read + filename, 0)

    if np.max(img)==0:
        image_filtered=img
    else:
        img1 = img.copy()
        # 计算非0连通域数量
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img1, connectivity=8)


        areas = stats[1:, cv2.CC_STAT_AREA]  # 忽略背景连通域
        max_area_index = np.argmax(areas) + 1  # 获取除了背景以外的最大连通域的索引
        max_area = areas[max_area_index - 1]  # 获取最大连通域的面积
        print(max_area)
        if nLabels <= 3 and max_area > 2000:  ####     图较大，且存在需要连接起来的
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        if nLabels > 3 and max_area > 2000:  ####       图较大，需要连接起来的较多
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
        if nLabels > 3 and max_area <= 2000:  ####      图较小
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if nLabels <= 3 and max_area <= 2000:  ####     图较小，且效果较好
            kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel1)
        # 查找轮廓并计算凸包
        contours, hierarchy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        remove_longest_segments(contours)
        img1 = FillHole(img1)
        image_filtered=img1

        if type(img1)==int:
            image_filtered=np.zeros_like(img)
        else:
            pass







    cv2.imwrite(path_gray + filename,image_filtered)
