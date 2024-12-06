import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

img_url = '/home/work/yaiq/YAIQ/workspace/SM/dataset_cut/vcog-bench/marsvqa/16/q1.jpg'
raw_image = Image.open(img_url).convert('RGB')

inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print('original blip inference')
print(processor.decode(out[0], skip_special_tokens=True))
