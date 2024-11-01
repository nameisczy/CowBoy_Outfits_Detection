import torch
import torchvision
import wandb
import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import yaml
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import subprocess
from PIL import Image

# 读取JSON文件
json_file_path = 'kaggle/CowBoy_Outfits_Detection/train.json'
data = json.load(open(json_file_path, 'r'))

# 创建YOLO注释文件目录
yolo_anno_path = 'kaggle/CowBoy_Outfits_Detection/yolo_anno/'
if not os.path.exists(yolo_anno_path):
    os.makedirs(yolo_anno_path)

# 创建类别ID映射
cate_id_map = {}
num = 0
for cate in data['categories']:
    cate_id_map[cate['id']] = num
    num += 1

# 将COCO格式的边界框转换为YOLO格式
def cc2yolo_bbox(img_width, img_height, bbox):
    dw = 1. / img_width
    dh = 1. / img_height
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# 创建train.csv并写入数据
f = open('kaggle/CowBoy_Outfits_Detection/train.csv', 'w')
f.write('id,file_name\n')
for i in tqdm(range(len(data['images']))):
    filename = data['images'][i]['file_name']
    img_width = data['images'][i]['width']
    img_height = data['images'][i]['height']
    img_id = data['images'][i]['id']
    yolo_txt_name = filename.split('.')[0] + '.txt'  # 移除.jpg

    f.write('{},{}\n'.format(img_id, filename))
    yolo_txt_file = open(os.path.join(yolo_anno_path, yolo_txt_name), 'w')

    for anno in data['annotations']:
        if anno['image_id'] == img_id:
            yolo_bbox = cc2yolo_bbox(img_width, img_height, anno['bbox'])  # "bbox": [x,y,width,height]
            yolo_txt_file.write(
                '{} {} {} {} {}\n'.format(cate_id_map[anno['category_id']], yolo_bbox[0], yolo_bbox[1], yolo_bbox[2],
                                          yolo_bbox[3]))
    yolo_txt_file.close()
f.close()

# 读取train.csv并进行数据集划分
train = pd.read_csv('kaggle/CowBoy_Outfits_Detection/train.csv')
train_df, valid_df = train_test_split(train, test_size=0.10, random_state=233)

train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'
df = pd.concat([train_df, valid_df]).reset_index(drop=True)

# 创建训练和验证集的图片和标签文件夹
os.makedirs('kaggle/CowBoy_Outfits_Detection/cowboy/images/train', exist_ok=True)
os.makedirs('kaggle/CowBoy_Outfits_Detection/cowboy/images/valid', exist_ok=True)
os.makedirs('kaggle/CowBoy_Outfits_Detection/cowboy/labels/train', exist_ok=True)
os.makedirs('kaggle/CowBoy_Outfits_Detection/cowboy/labels/valid', exist_ok=True)

# 将图片和标签文件复制到相应的训练和验证集文件夹中
for i in tqdm(range(len(df))):
    row = df.loc[i]
    name = row.file_name.split('.')[0]
    if row.split == 'train':
        copyfile(f'kaggle/CowBoy_Outfits_Detection/images/{name}.jpg', f'kaggle/CowBoy_Outfits_Detection/cowboy/images/train/{name}.jpg')
        copyfile(f'kaggle/CowBoy_Outfits_Detection/yolo_anno/{name}.txt', f'kaggle/CowBoy_Outfits_Detection/cowboy/labels/train/{name}.txt')
    else:
        copyfile(f'kaggle/CowBoy_Outfits_Detection/images/{name}.jpg', f'kaggle/CowBoy_Outfits_Detection/cowboy/images/valid/{name}.jpg')
        copyfile(f'kaggle/CowBoy_Outfits_Detection/yolo_anno/{name}.txt', f'kaggle/CowBoy_Outfits_Detection/cowboy/labels/valid/{name}.txt')

# 创建YOLOv5配置文件
data_yaml = dict(
    train = 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/cowboy/images/train',
    val = 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/cowboy/images/valid',
    nc = 5,
    names = ['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket']
)

# 将配置文件保存到指定路径
with open('C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)

# 开始训练
epochs = 5
batch_size = 32
model_type = 'yolov5m.pt'  # 选择模型类型，如yolov5s.pt, yolov5m.pt, yolov5l.pt


command = [
    'python', 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/yolov5/train.py',
    '--img', '640',  # 图片大小
    '--batch', str(batch_size),
    '--epochs', str(epochs),
    '--data', 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/yolov5/data/data.yaml',
    '--weights', model_type,
    '--name', 'cowboy_detection',
    '--save-period', '1',  # 每隔1个epoch保存一次
]

subprocess.run(command)
