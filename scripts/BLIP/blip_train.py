from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from torch.optim import AdamW
from tqdm import tqdm
import wandb

class CustomDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.dataframe = dataframe
        self.processor = processor
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['label']
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=label, images=image, return_tensors="pt", padding=True, truncation=True)
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
        }
        
def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item["input_ids"] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    return {
        "input_ids": input_ids.to(device),
        "pixel_values": pixel_values.to(device)
    }
    
wandb.init(project="yaiq_blip", config={"learning_rate":5e-6, "epochs": 25})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

data_path = "../csv/clip_dataset.csv"
df = pd.read_csv(data_path)
dataset = CustomDataset(dataframe=df, processor=processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-6)

model.train()
global_step = 0
for epoch in range(15):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        
        loss = outputs.loss
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({"loss": loss.item(), "epoch": epoch+1, "global_step": global_step})
        global_step += 1
        
        progress_bar.set_postfix({"loss": loss.item()})
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
    
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
    
    if (epoch+1) % 5 == 0:
        model.save_pretrained(f"../models/fine_tuned_blip_image_captioning_large_epoch_{epoch + 1}")
        processor.save_pretrained(f"../models/fine_tuned_blip_image_captioning_large_epoch_{epoch + 1}")
    
wandb.finish()