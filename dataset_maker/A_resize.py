import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_SIZE = 512
PIC_DIR = "B"
OUT_DIR = "RESIZE"  

file_list = os.listdir(PIC_DIR)
pic_count = 0
OUT_DIR = OUT_DIR + PIC_DIR

try:
    os.makedirs(OUT_DIR)
except :
    print ("路径已存在")

for file_name in file_list :
    try :
        img = cv.imread(PIC_DIR + "/" + file_name)

        #图像切割成正方形
        size = min([img.shape[0], img.shape[1]])
        img_cut = img[0: size-1, 0: size-1]
        img_resize = cv.resize(img_cut, (OUT_SIZE, OUT_SIZE))

        #拉普拉斯锐化
        kern = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        imgLaplace = cv.filter2D(img_resize, -1, kern)
        imgLaplace = np.float32(imgLaplace)

        #图像二值化与反色
        imgLaplace = np.heaviside(imgLaplace - 100, 0)
        imgLaplace = 254 * (1 - imgLaplace)
        imgLaplace = np.float32(imgLaplace)

        #拼接
        img_train = np.concatenate([img_resize, imgLaplace], axis=1)

        '''
        plt.imshow(img_train)
        plt.show()
        '''

        cv.imwrite(OUT_DIR + "/" + PIC_DIR + str(pic_count) + ".png", img_train)

        pic_count += 1
        
    except Exception as e:
        print(file_name, "处理失败，异常", e)

print("处理成功", pic_count, "张")
