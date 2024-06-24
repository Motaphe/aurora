import torch
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn
from io import BytesIO
import os
import pickle
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

temp = pd.read_excel(r"anime.xlsx")
classes = temp["Col_Names"].tolist()
classes = [s.lstrip() for s in classes]
positive_classes = []
negative_classes = []
for i in range(len(classes)):
    positive_classes.append(f"a outstanding picture of a {classes[i]}")
    negative_classes.append(f"a horrible picture of a {classes[i]}")

positive_inputs = processor(text=positive_classes, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    positive_text_features = model.get_text_features(**positive_inputs)
negative_inputs = processor(text=negative_classes, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    negative_text_features = model.get_text_features(**negative_inputs)

positive_prompt_vectors = np.array(positive_text_features)
average_positive_vector = np.mean(positive_prompt_vectors, axis=0)

negative_prompt_vectors = np.array(negative_text_features)
average_negative_vector = np.mean(negative_prompt_vectors, axis=0)

with open('anime_positive_prompt.pkl', 'wb') as f:
    pickle.dump(average_positive_vector, f)
with open('anime_negative_prompt.pkl', 'wb') as f:
    pickle.dump(average_negative_vector, f)

