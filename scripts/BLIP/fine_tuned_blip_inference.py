import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
import json

fine_tuned_model_path = "../models/fine_tuned_blip_image_captioning_large_epoch_10"
processor = BlipProcessor.from_pretrained(fine_tuned_model_path)
model = BlipForConditionalGeneration.from_pretrained(fine_tuned_model_path).to("cuda")

df = pd.read_csv('../csv/MENSA_Norway.csv')

rows = []
prompt = {}

for i in range(1, 36):
    question_directory = f'/home/work/yaiq/YAIQ/workspace/SM/dataset_cut/MENSA/train/Q{i}'
    question_texts = []
    for j in range(0, 8):
        question_path = question_directory + f'/q{j}.jpg'
        raw_image = Image.open(question_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt", max_length=4096, truncation=True).to("cuda")
        out = model.generate(**inputs)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        question_texts.append(text)
        print(text)
        new_row = {'index': i, 'image_path': question_path, 'number': f'q{j}', 'text': text}
        rows.append(new_row)
        
    option_directory = f'/home/work/yaiq/datasets/MENSA/train/Q{i}'
    option_texts = []
    for letter in ['A', 'B', 'C', 'D', 'E', 'F']:
        option_path = option_directory + f'/{letter}.png'
        raw_image = Image.open(option_path).convert('RGB')
        inputs = processor(raw_image, return_tensors="pt", max_length=4096, truncation=True).to("cuda")
        out = model.generate(**inputs)
        text = processor.batch_decode(out, skip_special_tokens=True)[0]
        option_texts.append(text)
        print(text)
        new_row = {'index': i, 'image_path': option_path, 'number': f'{letter}', 'text': text}
        rows.append(new_row)
        
    basic_prompt = f"""
You will be given a series of 8 images described in text. After reading the descriptions of the 8 images, you will then be presented with 6 possible image options. Your task is to choose the image that best represents what the next image in the sequence would be based on the descriptions provided.

Here are the descriptions of the 8 images:

Image 1: {question_texts[0]}
Image 2: {question_texts[1]}
Image 3: {question_texts[2]}
Image 4: {question_texts[3]}
Image 5: {question_texts[4]}
Image 6: {question_texts[5]}
Image 7: {question_texts[6]}
Image 8: {question_texts[7]}
Based on the sequence of these images, choose the image that best represents the next one in the series. Here are the 6 options:

Option A: {option_texts[0]}
Option B: {option_texts[1]}
Option C: {option_texts[2]}
Option D: {option_texts[3]}
Option E: {option_texts[4]}
Option F: {option_texts[5]}
Please respond with the letter corresponding to the image you think follows the sequence.
    """
    
    inside = {
        "prompt" : basic_prompt,
        "correct_answer" : df.iloc[i-1]['correct_answer']
    }
    
    prompt[f"Question {i}"] = inside
        
    print(f'index: {i}')
        
df = pd.DataFrame(rows)
df.to_csv('../csv/mensa_dataset_with_blip_inference.csv', index=False)
with open('../mensa_prompt.json', 'w') as json_file:
    json.dump(prompt, json_file, indent=4)

out = model.generate(**inputs)
print(out)
print(processor.batch_decode(out, skip_special_tokens=True))
