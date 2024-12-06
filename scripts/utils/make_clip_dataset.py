from PIL import Image
import os
import pandas as pd
import json

default_dataset_path = "/home/work/yaiq/datasets"
preprocess_dataset_path = "/home/work/yaiq/YAIQ/workspace/SM/dataset_cut"

rows = []

# raven
for i in range(560):
    question_dir = os.path.join('/home/work/yaiq/datasets/vcog-bench/raven', str(i), 'choice')
    json_file = os.path.join(question_dir, 'text', 'annotation.json')
    
    with open(json_file, 'r') as file:
        annotation = json.load(file)
    
    if (i % 11 == 0 and i <= 539):
        split = 'test'
    else:
        split = 'train'
        
    choices = os.listdir(question_dir + '/image')
    
    for choice in choices:
        image_path = os.path.join(question_dir, 'image', choice)
        label = annotation[choice]
        new_row = {'image_path': image_path, 'label': label, 'split': split}
        rows.append(new_row)
        
# mars
for i in range(480):
    question_dir = os.path.join('/home/work/yaiq/datasets/vcog-bench/marsvqa', str(i), 'choice')
    json_file = os.path.join(question_dir, 'text', 'annotation.json')
    
    with open(json_file, 'r') as file:
        annotation = json.load(file)
    
    if (i % 9 == 0 and i <= 441):
        split = 'test'
    else:
        split = 'train'
        
    choices = os.listdir(question_dir + '/image')
    
    for choice in choices:
        image_path = os.path.join(question_dir, 'image', choice)
        label = annotation[choice]
        new_row = {'image_path': image_path, 'label': label, 'split': split}
        rows.append(new_row)
        
df = pd.DataFrame(rows)
df.to_csv('../csv/clip_dataset.csv', index=False)