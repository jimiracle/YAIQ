from transformers import BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
import torch
import av
import numpy as np
from huggingface_hub import hf_hub_download
import os
import cv2
import json
import pandas as pd

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype="float16", device_map='auto')
processor = LlavaOnevisionProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
processor.tokenizer.padding_side = "left"

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_images(image_paths, max_frames_num):
    file_paths = []
    
    for root, dirs, files in os.walk(image_paths):
        for file in files:
            file_paths.append(os.path.join(root, file))

    sampled_images = []
    
    sampled_image_paths = file_paths[:max_frames_num] 

    for img_path in sampled_image_paths:
        img = cv2.imread(img_path)
        
        if img is not None:
            sampled_images.append(img)
        else:
            print(f"Warning: Unable to load image at {img_path}")
    
    return np.array(sampled_images)

default_dataset_path = "/home/work/yaiq/datasets"
csv_path = "/home/work/yaiq/datasets/dataset50.csv"
preprocess_dataset_path = "/home/work/yaiq/YAIQ/workspace/SM/dataset_cut"
data = pd.read_csv(csv_path)
results = []

for idx, row in data.iterrows():
    if row['dataset'] == 'mars':
        images_directory = os.path.join(preprocess_dataset_path, 'vcog-bench', 'marsvqa', str(row['index']))
        answer_image_directory = os.path.join(default_dataset_path, 'vcog-bench', 'marsvqa', str(row['index']), 'answer', 'image')
        text_path = os.path.join(default_dataset_path, 'vcog-bench', 'marsvqa', str(row['index']), 'choice', 'text', 'annotation.json')
        
        answer = os.listdir(answer_image_directory)[0]

    else:
        images_directory = os.path.join(preprocess_dataset_path, row['path'])
        answer_image_directory = os.path.join(default_dataset_path, row['path'], 'answer', 'image')
        text_path = os.path.join(default_dataset_path, row['path'], 'choice', 'text', 'annotation.json')

        answer = os.listdir(answer_image_directory)[0]
        
    video_frames = load_images(images_directory, 8)
    print(video_frames.shape)

    with open(text_path, 'r') as file:
        texts = json.load(file)

    bogi_file_names = []
    bogi_texts = []
    for key, value in texts.items():
        # if row['dataset'] == 'raven':
        #     bogi_file_names.append(key[:-5])
        # else:
        #     bogi_file_names.append(key.split("_")[2][1:])
        bogi_file_names.append(key)
        bogi_texts.append(value)
        
    if row['dataset'] == 'raven':
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You have the following 8 options to choose from. Each option includes a brief description and a corresponding filename. Based on the descriptions, select one option for the next frame that follows the structural and analogical relations. Please select by replying with the option number. Do not include any explanation or reasoning in your response.\nOptions:\n1. {bogi_texts[0]}\n\tFilename: {bogi_file_names[0]}\n2. {bogi_texts[1]}\n\tFilename: {bogi_file_names[1]}\n3. {bogi_texts[2]}\n\tFilename: {bogi_file_names[2]}\n4. {bogi_texts[3]}\n\tFilename: {bogi_file_names[3]}\n5. {bogi_texts[4]}\n\tFilename: {bogi_file_names[4]}\n6. {bogi_texts[5]}\n\tFilename: {bogi_file_names[5]}\n7. {bogi_texts[6]}\n\tFilename: {bogi_file_names[6]}\n8. {bogi_texts[7]}\n\tFilename: {bogi_file_names[7]}\n"},
                    {"type": "video"},
                    ],
            },
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You have the following 4 options to choose from. Each option includes a brief description and a corresponding filename. Based on the descriptions, select one option for the next frame that follows the structural and analogical relations. Please select by replying with the option number. Do not include any explanation or reasoning in your response.\nOptions:\n1. {bogi_texts[0]}\n\tFilename: {bogi_file_names[0]}\n2. {bogi_texts[1]}\n\tFilename: {bogi_file_names[1]}\n3. {bogi_texts[2]}\n\tFilename: {bogi_file_names[2]}\n4. {bogi_texts[3]}\n\tFilename: {bogi_file_names[3]}\n"},
                    {"type": "video"},
                    ],
            },
        ]
        
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, videos=video_frames, padding=True, return_tensors="pt").to(model.device, torch.float16)
    generate_kwargs = {"max_new_tokens": 100, "do_sample": True, "top_p": 0.9}

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)

    print(generated_text[0])
    results.append({
            "image_directory": images_directory,
            "text_prompt": prompt,
            "output": generated_text[0],
            "answer": answer,
        })
    
output_path = "results_onevision_video.json"
with open(output_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Results saved to {output_path}")