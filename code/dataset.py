from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import os
from utils import generate_mask
import numpy as np



class FoodDataset(Dataset):
    
    def __init__(self, config, df, transform):
        """
        img_dir: is the directory to image folder
        df: is the dataframe annotation image and attribute
        transform: transform augmentation
        """
        self.df = df
        self.config = config
        self.transform = transform

    def __getitem__(self, idx):
        img_path = os.path.join(self.config.DATA.PATH, self.df[self.config.DATA.INPUT_NAME].iloc[idx])
        annos = self.df[self.config.DATA.TARGET_NAME].iloc[idx]
        res = (self.df['width'].iloc[idx], self.df['height'].iloc[idx])
        mask = generate_mask(annos, res)
        mask = Image.fromarray(mask)
        img = Image.open(img_path)
        # img = self.transform(img)
        if self.transform:
            img = self.transform['image'](img)
            mask = self.transform['mask'](mask)
        mask = torch.FloatTensor(np.expand_dims(np.array(mask),axis=0))
        return img, mask
    
    def __len__(self):
        return len(self.df)