import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MyLabeledDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, transform_mask=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.transform_mask = transform_mask
        
        self.images = sorted([
            os.path.join(image_dir, x) 
            for x in os.listdir(image_dir) 
            if x.endswith('.png') or x.endswith('.jpg')
        ])
        
        self.masks = sorted([
            os.path.join(mask_dir, x) 
            for x in os.listdir(mask_dir) 
            if x.endswith('.png') or x.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")  # or "L" if grayscale

        # Load corresponding mask
        mask_path = self.masks[idx]
        mask = Image.open(mask_path).convert("L")  # single-channel

        if self.transform:
            image = self.transform(image)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        # Binarize the mask (assuming foreground is > 0.5)
        mask = (mask > 0.5).long()  
        mask = mask.squeeze(0)     

        return image, mask

class MyUnlabeledDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([
            os.path.join(image_dir, x) 
            for x in os.listdir(image_dir) 
            if x.endswith('.png') or x.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image
