# -*- coding: utf-8 -*-
# Inference with Video-LLaVa
## Set-up environment
"""
pip install --upgrade -q accelerate bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install -q av
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = '1'

import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
from huggingface_hub import hf_hub_download
import av
import numpy as np
import json


print(torch.cuda.is_available())

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

image_paths = ['./demo/0_resize/q0.jpg', './demo/0_resize/q1.jpg', './demo/0_resize/q2.jpg', './demo/0_resize/q3.jpg', './demo/0_resize/q4.jpg', './demo/0_resize/q5.jpg', './demo/0_resize/q6.jpg', './demo/0_resize/q7.jpg']

with open('/home/work/yaiq/datasets/vcog-bench/raven/0/choice/text/annotation.json', 'r') as file:
    texts = json.load(file)

bogi_texts = []
for text in texts:
    bogi_texts.append(texts[text])

model_id = "LanguageBind/Video-LLaVA-7B-hf"

processor = VideoLlavaProcessor.from_pretrained(model_id)
model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

clip = np.stack([np.array(Image.open(img_path).convert("RGB")) for img_path in image_paths])

print("Clip shape:", clip.shape)

prompt = f"""
USER: <video>
You are provided with the first eight elements in eight frames.
Please select one from eight choices following the structural and analogical relations.
A. {bogi_texts[0]}
B. {bogi_texts[1]}
C. {bogi_texts[2]}
D. {bogi_texts[3]}
E. {bogi_texts[4]}
F. {bogi_texts[5]}
G. {bogi_texts[6]}
H. {bogi_texts[7]}
ASSISTANT:
"""

inputs = processor(prompt, videos=clip, return_tensors="pt").to(model.device)
for k, v in inputs.items():
    print(k, v.shape)

generate_kwargs = {"max_new_tokens":8, "do_sample": True, "top_p": 0.9, "top_k": 2}

output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(generated_text[0])



image = Image.open('/home/work/yaiq/datasets/vcog-bench/raven/0/question/image/question.jpeg')

prompt = f"""
USER: <image>
You are provided with the first eight elements in eight frames.
Please select one from eight choices following the structural and analogical relations.
A. {bogi_texts[0]}
B. {bogi_texts[1]}
C. {bogi_texts[2]}
D. {bogi_texts[3]}
E. {bogi_texts[4]}
F. {bogi_texts[5]}
G. {bogi_texts[6]}
H. {bogi_texts[7]}
ASSISTANT:
"""
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

generate_ids = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(generated_text[0])