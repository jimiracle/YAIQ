import os
import pandas as pd

default_dataset_path = "/home/work/yaiq/datasets"
preprocess_dataset_path = "/home/work/yaiq/YAIQ/workspace/SM/dataset_cut"

rows = []

# marsvqa
for i in range(480):
    directory_index = i
    
    question_dir = os.path.join(preprocess_dataset_path, 'vcog-bench/marsvqa', str(i))
    questions = sorted(os.listdir(question_dir))[:-1]
    for question in questions:
        image_path = os.path.join(question_dir, question)
        new_row = {'index': i, 'image_path': image_path, 'number': question}
        rows.append(new_row)
    
    choice_dir = os.path.join('/home/work/yaiq/datasets/vcog-bench/marsvqa', str(i), 'choice')    
    choices = sorted(os.listdir(choice_dir + '/image'))
    
    for choice in choices:
        image_path = os.path.join(choice_dir, 'image', choice)
        new_row = {'index': i, 'image_path': image_path, 'number': choice}
        rows.append(new_row)
        
df = pd.DataFrame(rows)
df.to_csv('../csv/marsvqa_dataset.csv', index=False)