import cv2
import os
import math
import numpy as np
import sys
from sklearn import svm
from sklearn.externals import joblib
from ML.svm_glcm.svm_image_featuretr import testfeature,reRGBandHLS
from ML.svm_glcm.Spectral_Feature import Spectral_Features

sys.path.append("..")
from base.base import interface
#垃圾桶满溢分类
class HandleGarbage(interface):
    def __init__(self):
        # 提取模型
        self.checkpoint = joblib.load('ML/weights/svm_garbage.model')     
        self.Feature_Color = Spectral_Features()
        self.data=np.zeros((1, 16))  # 标签（0 空 1 满）
    def inference(self, img):
        img = img[0:img.shape[0]//3 , ...]
        #cv2.imwrite('result.jpg', img)
        asm, con, eng, idm = testfeature(img)        
        #添加4维灰度共生矩阵特征 角二阶矩(能量)，对比度，熵，反差分矩阵(逆方差)
        self.data[0, 0:4]=asm, con, eng, idm
        # 4-9 添加6类图像空间特征 R G B H L S
        self.data[0, 4:10]=reRGBandHLS(img)        
        #添加 6类图像特征 标准差 偏差 R-G R-B G-B 覆盖率
        img = cv2.resize(img, (250, 300), interpolation=cv2.INTER_AREA)
        other_feature = np.array(self.Feature_Color.Cal_SpetrelFeature(img)).astype('double')
        for j in range(len(other_feature)-2):
            self.data[0][j+10] = other_feature[j+2]

        y_pred = self.checkpoint.predict(self.data)
        return y_pred


#return 0为空, 1为满
def waste_container_status(images, bbox):
    ret = []
    svm_model = HandleGarbage()

    if images is None:
        return -1

    for index, box in enumerate(bbox):

        lefttop = box[0]
        rightbottom = box[2]

        image= images[lefttop[1]:rightbottom[1], lefttop[0]:rightbottom[0]]
        #cv2.imwrite('result.jpg', image)
        if image is None:
            return -1
        #print(svm_model.inference(image)[0])
        status = svm_model.inference(image)
        #print(status)
        if status[index]:
            ret.append(index)
        
    return ret

if __name__ == '__main__':

    #img = cv2.imread('../images/02.png')
    #bbox = [[444,246,583,397], [650,265,754,406], [749,257,861,417] ]#3 trash position [left,top, right, bottom ]

    img = cv2.imread('images/test.jpg')
    bbox = [ [ [918,508], [1128,508], [1128, 799], [918,799] ] ]
    print(waste_container_status(img, bbox))


