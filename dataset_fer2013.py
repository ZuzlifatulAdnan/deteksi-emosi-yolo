import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class FER2013Dataset(Dataset):
    def __init__(self, csv_path, split="Training", transform=None, target_size=48):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["Usage"] == split].reset_index(drop=True)
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = np.array(list(map(int, row["pixels"].split())), dtype=np.uint8)
        img = pixels.reshape(48, 48)  # grayscale 48x48
        label = int(row["emotion"])

        # resize ke target_size dan buat 3-channel untuk CNN umum
        img = cv2.resize(img, (self.target_size, self.target_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]
        else:
            # Normalisasi [0,1] & to tensor CHW
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label
