from PIL import Image
import os
import pandas as pd

def split_image_into_9_parts(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(image_path)
    width, height = image.size
    
    part_width = width // 3
    part_height = height // 3
    
    count = 0
    for row in range(3):
        for col in range(3):
            left = col * part_width + 10
            upper = row * part_height + 10
            right = (col + 1) * part_width - 10
            lower = (row + 1) * part_height - 10
            
            cropped_image = image.crop((left, upper, right, lower))
            if cropped_image.mode == 'RGBA':
                cropped_image = cropped_image.convert('RGB')
            
            part_filename = os.path.join(output_dir, f"q{count}.jpg")
            
            cropped_image.save(part_filename)
            print(f"Saved {part_filename}")
            count += 1
    
    
default_dataset_path = "/home/work/yaiq/datasets"
preprocess_dataset_path = "/home/work/yaiq/YAIQ/workspace/SM/dataset_cut"
    
# raven image crop
for i in range(0, 560):
    path = 'vcog-bench/raven/' + str(i)
    image_directory = os.path.join(default_dataset_path, path)
    cut_image_directory = os.path.join(preprocess_dataset_path, path)
    
    if not os.path.exists(cut_image_directory):
        os.makedirs(cut_image_directory)
        print(f"Directory created: {cut_image_directory}")
    else:
        print(f"Directory already exists: {cut_image_directory}")
    
    question_image_directory = os.path.join(image_directory, 'question', 'image')
    
    items = os.listdir(question_image_directory)
    files = [os.path.join(question_image_directory, item) for item in items if os.path.isfile(os.path.join(question_image_directory, item))]
    question_path = files[0]
    
    split_image_into_9_parts(question_path, cut_image_directory)
    
# mars image crop

for i in range(0, 480):
    path = 'vcog-bench/marsvqa/' + str(i)
    image_directory = os.path.join(default_dataset_path, path)
    cut_image_directory = os.path.join(preprocess_dataset_path, path)
    
    if not os.path.exists(cut_image_directory):
        os.makedirs(cut_image_directory)
        print(f"Directory created: {cut_image_directory}")
    else:
        print(f"Directory already exists: {cut_image_directory}")
    
    question_image_directory = os.path.join(image_directory, 'question', 'image')
    
    items = os.listdir(question_image_directory)
    files = [os.path.join(question_image_directory, item) for item in items if os.path.isfile(os.path.join(question_image_directory, item))]
    question_path = files[0]
    
    split_image_into_9_parts(question_path, cut_image_directory)


# MENSA image crop

for i in range(1, 36):
    path = 'MENSA/train/Q' + str(i)
    image_directory = os.path.join(default_dataset_path, path)
    cut_image_directory = os.path.join(preprocess_dataset_path, path)
    
    if not os.path.exists(cut_image_directory):
        os.makedirs(cut_image_directory)
        print(f"Directory created: {cut_image_directory}")
    else:
        print(f"Directory already exists: {cut_image_directory}")
    
    question_path = os.path.join(image_directory, 'Q.png')
    
    split_image_into_9_parts(question_path, cut_image_directory)