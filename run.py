import cv2
import mmcv
from ML.interface import waste_container_status as minterface
from DL.interface import waste_container_status as diterface


if __name__ == '__main__':

    #ML方法调用接口，传入单张垃圾桶图片 垃圾桶框为空
    img = cv2.imread('images/test.jpg')
    bbox = [ [ [918,508], [1128,508], [1128, 799], [918,799] ] ]
    print(minterface(img, bbox))


    #DL 方法调用接口 目前传入大图，以及垃圾桶框位置, 若预先标记框的坐标，则不需要使用此方法
    img = cv2.imread('images/test.jpg')
    bbox = [ [ [918,508], [1128,508], [1128, 799], [918,799] ] ]
    print(diterface(img, bbox))