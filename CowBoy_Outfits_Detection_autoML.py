import numpy as np
import pandas as pd
import logging
import sys
import os
import json
import zipfile
from pycocotools.coco import COCO
from PIL import Image

from autogluon.multimodal import MultiModalPredictor

# Setting up the logger
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))

# Root path to the data
root = 'kaggle/CowBoy_Outfits_Detection'

# Load the COCO dataset
coco = COCO(os.path.join(root, 'train.json'))

# Print dataset information and categories
print('Data info:', coco.info())
categories = {cat_info['name']: cat_info['id'] for cat_info in coco.loadCats(coco.getCatIds())}
print('Categories:', categories)

# Prepare the training data
train_df = pd.DataFrame.from_dict(coco.dataset['images'])
annotations = pd.DataFrame.from_dict(coco.dataset['annotations'])
train_df = train_df.merge(annotations, left_on='id', right_on='image_id')

# Modify the file paths to be absolute paths
train_df['file_name'] = train_df['file_name'].apply(lambda x: os.path.join(root, 'images', x))

# Rename 'file_name' to 'image' to match expected format
train_df.rename(columns={'file_name': 'image'}, inplace=True)

# Convert 'bbox' to 'rois' column format
train_df['rois'] = train_df['bbox'].apply(lambda x: [{'xmin': x[0], 'ymin': x[1], 'xmax': x[0] + x[2], 'ymax': x[1] + x[3]}])

# Add 'label' column corresponding to 'category_id'
train_df['label'] = train_df['category_id']

# Drop unnecessary columns (optional)
train_df = train_df[['image', 'rois', 'label']]

# Randomly select 10 images for each category as validation data
sample_n_per_cat = 10
valid_ids = pd.Index([], dtype='int64')
for cat_name in categories.keys():
    df = train_df[train_df['label'] == categories[cat_name]]
    df = df.sample(sample_n_per_cat)
    valid_ids = valid_ids.append(df.index)

train_ids = train_df.index.drop(valid_ids)
train_data = train_df.loc[train_ids]
valid_data = train_df.loc[valid_ids]

print('Train split:', len(train_data), 'Validation split:', len(valid_data))

# Train the model using MultiModalPredictor
predictor = MultiModalPredictor(label='label', problem_type='object_detection')

# Removed 'model.backbone' from hyperparameters, using only supported parameters
predictor.fit(
    train_data=train_data,
    tuning_data=valid_data,
    hyperparameters={
        'optimization.max_epochs': 3
    }
)

# Function to create a submission file
def create_submission(df, predictor, score_thresh=0.1):
    results = []
    for index, row in df.iterrows():
        img_id = row['image_id']
        file_name = row['file_name']
        img = Image.open(file_name)
        width, height = img.size
        output = predictor.predict({'image': file_name})
        for _, p in output.iterrows():
            if p['predict_score'] > score_thresh:
                roi = p['predict_rois']
                pred = {'image_id': img_id,
                        'category_id': categories[p['predict_class']],
                        'bbox': [roi['xmin'] * width, roi['ymin'] * height, roi['xmax'] * width, roi['ymax'] * height],
                        'score': p['predict_score']}
                results.append(pred)
    return results

# Prepare submission
submission_df = pd.read_csv(os.path.join(root, 'valid.csv'))  # replace with test.csv on the last day
submission_df['file_name'] = submission_df.apply(lambda x: os.path.join(root, 'images', x['file_name']), axis=1)
submission = create_submission(submission_df, predictor)

# Save the submission file
submission_name = '/kaggle/working/answer.json'
with open(submission_name, 'w') as f:
    json.dump(submission, f)
zf = zipfile.ZipFile('/kaggle/working/sample_answer.zip', 'w')
zf.write(submission_name, 'answer.json')
zf.close()
