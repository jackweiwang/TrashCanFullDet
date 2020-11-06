from abc import ABCMeta, abstractmethod


#抽象类
class interface(object):
    @abstractmethod  
    def __init__(self, **kwargs):
        pass
    def inference(self, image):
        pass
