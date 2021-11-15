
 
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

优劣比较  
1.优点深度学习方法 提升空间大增加样本，调整网络必然会增加map 适合使用在无法预估环境或环境变化明显的场景， 可检测分割一起测试不需要定位， 缺点：需要增加样本，迭代较慢。
2.优点传统方法速度快的方法和慢的方法 无明显精度差距，迭代快，训练速度快，使用方便 目前维持精度85，缺点精度很难上升。

需要你的星星 # - -
