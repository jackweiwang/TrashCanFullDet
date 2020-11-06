
import matplotlib.pyplot as plt
import mmcv
import cv2
import sys
import numpy as np
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
sys.path.append("..")
from base.base import interface
#垃圾分割类
class HandleGarbage(interface):
    def __init__(self):

        config = 'DL/configs/hrnet/mask_rcnn_hrnetv2p_w32_2x_coco.py'
        checkpoint = 'DL/weights/mask_rcnn_hrnetv2p_w32_2x_coco/model.pth'
        device ='cuda:0'
        self.model = init_detector(config, checkpoint, device)
        if hasattr(self.model, 'module'):
            self.model = self.model.module

    def inference(self, image):

        return inference_detector(self.model, image)


#返回list为输入框的个数， -1代表未检测到垃圾桶， 0代表检测到垃圾桶但未检测到垃圾， 大于0的数为垃圾的面积
def waste_container_status(images, bbox):

    ret = []
    for index, box in enumerate(bbox):
        #print(box)
        lefttop = box[0]
        rightbottom = box[2]
        image= images[lefttop[1]:rightbottom[1], lefttop[0]:rightbottom[0]]

        result = HandleGarbage().inference(image)
        img = image.copy()

        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        #print(labels)
        labels = np.concatenate(labels)
        # draw segmentation masks
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            #inds = np.where(bboxes[:, -1] > 0.4)[0]
            np.random.seed(42)
            # for i in inds:
            #     i = int(i)

            mask = segms[0]

            img[mask] = img[mask]*0.0 
            if np.sum(mask):
                ret.append(index)

    return ret

#demo 测试接口可参考该函数调用接口
if __name__ == '__main__':
    img = mmcv.imread('../images/02.png')
    bbox = [[444,246,583,397], [650,265,754,406], [749,257,861,417] ]#3 trash position [left,top, right, bottom ]
    print(waste_container_status(img, bbox))
