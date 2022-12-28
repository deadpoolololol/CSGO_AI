# CSGO_AI

使用 yolov3 模型训练 csgo 人头检测并实现锁头功能

## 数据集

下载链接：https://pan.baidu.com/s/16fXR9hSovy350WVI7_TN7Q?pwd=6666 
提取码：6666 

### 介绍

分为 CT 和 T 两类

包括 jpg 文件和 xml 文件

将所有 jpg 文件导入到 VOCdevkit/VOC2007/JPEGImages 中

将所有 xml 文件导入到 VOCdevkit/VOC2007/Annotations 中

## 环境需求

主目录中打开终端输入

```
pip install -r requirements.txt
```

### 权重下载

下载链接：https://pan.baidu.com/s/16-2X3lPGi2pXF71oNn2TDA?pwd=6666 
提取码：6666 

## 使用方法

将权重文件 best.pt 放在主目录

直接运行 detect_CSGO_AI.py 文件

等模型加载完毕后选择需要检测的阵营

单击右键即可实现锁头

## 训练方法

参考 [(79条消息) yolov3模型训练——使用yolov3训练自己的模型_萝北村的枫子的博客-CSDN博客_yolov3训练模型](https://blog.csdn.net/thy0000/article/details/124579443)

## 注意事项

由于训练数据集分辨度问题,模型在适用于中近距离大部分探员识别,远距离识别效果欠佳

