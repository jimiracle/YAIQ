from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fine_tuned_model_path = "./fine_tuned_clip"
fine_tuned_model_path = "../models/fine_tuned_clip_large_patch14"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

image_path = "/home/work/yaiq/YAIQ/workspace/SM/dataset_cut/vcog-bench/raven/1/q2.jpg"
image = Image.open(image_path).convert("RGB")

# labels = ['Circle', 'Triangle', 'Square', 'Pentagon', 'Hexagon']
# labels = ['One Hexagon', 'Two Hexagon', 'Three Hexagon', 'Four Hexagon', 'Five Hexagon', 'Six Hexagon', 'Seven Hexagon', 'Eight Hexagon', 'Nine Hexagon']
labels = ['Center', '2x2 Grid', '3x3 Grid', 'Left-Right', 'Up-Down', 'Out-InCenter', 'Out-InGrid']

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval()

final_labels = []
best_label = None
best_prob = 0.0

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

print("Similarity score (softmax probabilities):", probs)
max_index = torch.argmax(probs, dim=1).item()

selected_label = labels[max_index]
print(f"Selected label in original CLIP: {selected_label}")