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

pred_path = 'kaggle/CowBoy_Outfits_Detection/yolov5/runs/detect/exp/labels/'

prediction_files = os.listdir(pred_path)

def yolo2cc_bbox(img_width, img_height, bbox):
    x = (bbox[0] - bbox[2] * 0.5) * img_width
    y = (bbox[1] - bbox[3] * 0.5) * img_height
    w = bbox[2] * img_width
    h = bbox[3] * img_height

    return (x, y, w, h)

re_cate_id_map = dict(zip(cate_id_map.values(), cate_id_map.keys()))

print(re_cate_id_map)

def make_submission(df, PRED_PATH, IMAGE_PATH):
    output = []
    for i in tqdm(range(len(df))):
        row = df.loc[i]
        image_id = row['id']
        file_name = row['file_name'].split('.')[0]
        if f'{file_name}.txt' in prediction_files:
            img = Image.open(f'{IMAGE_PATH}/{file_name}.jpg')
            width, height = img.size
            with open(f'{PRED_PATH}/{file_name}.txt', 'r') as file:
                for line in file:
                    preds = line.strip('\n').split(' ')
                    preds = list(map(float, preds)) #conver string to float
                    cc_bbox = yolo2cc_bbox(width, height, preds[1:-1])
                    result = {
                        'image_id': image_id,
                        'category_id': re_cate_id_map[preds[0]],
                        'bbox': cc_bbox,
                        'score': preds[-1]
                    }

                    output.append(result)
    return output

sub_data = make_submission(valid_df, pred_path, image_path)

op_pd = pd.DataFrame(sub_data)

op_pd.sample(10)

import zipfile

op_pd.to_json('kaggle/working/answer.json',orient='records')
zf = zipfile.ZipFile('kaggle/working/sample_answer.zip', 'w')
zf.write('kaggle/working/answer.json', 'answer.json')
zf.close()