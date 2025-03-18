
 
基于mmdetection框架 和sklearn 开发的垃圾桶满溢检测程序 

Setup  
pip install -r requirements.txt

切换环境source activate open-mmlab


Demo  
python run.py

Method  
传统机器学习：SVM glcm特征 rgb hls 标准差 偏差 特征

深度学习：hrnet特征提取 cascade maskrcnn 目标检测
测试结果测试10张垃圾桶图片， 测试结果为8张正确
测试100张 75左右

测试结果  
传统机器学习：144张图片  桶的数量不一， 总共测试结果ap 84 左右
深度学习：400左右样本训练模型，大图测试精度ap 75左右


感谢随手给个星星
