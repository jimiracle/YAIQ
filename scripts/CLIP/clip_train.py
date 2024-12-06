from transformers import CLIPProcessor, CLIPModel
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torch.optim import AdamW
from tqdm import tqdm
import wandb

wandb.init(project="yaiq", config={"learning_rate":5e-6, "epochs": 25})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

data_path = "../csv/clip_dataset.csv"
df = pd.read_csv(data_path)

dataset = CustomDataset(dataframe=df, processor=processor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=5e-6)


model.train()
global_step = 0
for epoch in range(50):
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        pixel_values = batch['pixel_values'].to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, return_loss=True)
        
        loss = outputs.loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({"loss": loss.item(), "epoch": epoch, "global_step": global_step})
        global_step += 1
        
        progress_bar.set_postfix({"loss": loss.item()})
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")
    
    wandb.log({"epoch_loss": avg_loss, "epoch": epoch})
    
    if epoch % 10 == 0:
        model.save_pretrained(f"../models/fine_tuned_clip_large_patch14_epoch_{epoch + 1}")
        processor.save_pretrained(f"../models/fine_tuned_clip_large_patch14_epoch_{epoch + 1}")
    
wandb.finish()