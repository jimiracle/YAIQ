import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        index = row['index']
        number = row['number']
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True)
        
        return {
            "index": index,
            "number": number,
            "image_path": image_path,
            "pixel_values": inputs["pixel_values"].squeeze(0)  # 배치 차원 제거
        }
        
def collate_fn(batch):
    indices = [item["index"] for item in batch]
    numbers = [item["number"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "indices": indices,
        "image_paths": image_paths,
        "numbers": numbers,
        "pixel_values": pixel_values.to("cuda")
    }

fine_tuned_model_path = "../models/fine_tuned_blip_image_captioning_large_epoch_10"
processor = BlipProcessor.from_pretrained(fine_tuned_model_path)
model = BlipForConditionalGeneration.from_pretrained(fine_tuned_model_path).to("cuda")

# data_path = "../csv/raven_dataset.csv"
data_path = "../csvmarsvqa_dataset.csv"
df = pd.read_csv(data_path)
dataset = CustomDataset(dataframe=df, processor=processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

rows = []

model.eval()
with torch.no_grad():
    for batch in dataloader:
        #input_ids = batch['input_ids'].to("cuda")
        print('batch paths')
        print(batch['image_paths'])
        pixel_values = batch['pixel_values'].to("cuda")

        outputs = model.generate(pixel_values=pixel_values)
        decode_list = processor.batch_decode(outputs, skip_special_tokens=True)
        print(processor.batch_decode(outputs, skip_special_tokens=True))
        
        for i in range(8):
            new_row = {'index': batch['indices'][i], 'image_path': batch['image_paths'][i], 'number': batch['numbers'][i], 'text': decode_list[i]}
            rows.append(new_row)

df = pd.DataFrame(rows)
# df.to_csv('../csv/raven_dataset_with_blip_inference.csv', index=False)
df.to_csv('../csv/marsvqa_dataset_with_blip_inference.csv', index=False)