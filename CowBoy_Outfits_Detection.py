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

valid_df = pd.read_csv('kaggle/CowBoy_Outfits_Detection/valid.csv')
test_df = pd.read_csv('kaggle/CowBoy_Outfits_Detection/test.csv')

# make directory to store the validation data.
os.makedirs('kaggle/CowBoy_Outfits_Detection/inference/valid', exist_ok=True)
os.makedirs('kaggle/CowBoy_Outfits_Detection/inference/test', exist_ok=True)

# copy the validation image to inference folder for detection process
for i in tqdm(range(len(valid_df))):
    row = valid_df.loc[i]
    name = row.file_name.split('.')[0]
    copyfile(f'kaggle/CowBoy_Outfits_Detection/images/{name}.jpg', f'kaggle/CowBoy_Outfits_Detection/inference/valid/{name}.jpg')

valid_path = 'kaggle/CowBoy_Outfits_Detection/inference/valid/'
model_path = 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/yolov5/runs/train/cowboy_detection4/weights/best.pt'
image_path = 'kaggle/CowBoy_Outfits_Detection/images/'

command = [
        'python', 'C:/Users/chenz/PycharmProjects/pythonProject/kaggle/CowBoy_Outfits_Detection/yolov5/detect.py',
        '--weights', model_path,
        '--source', valid_path,
        '--conf', '0.546',
        '--iou-thres', '0.5',
        '--save-txt',
        '--save-conf',
        '--augment'
    ]

subprocess.run(command)
