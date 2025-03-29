# utils/dataset_utils.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class RiverSegmentationDataset(Dataset):
    def __init__(self, csv_file, root_dir, rgb_folder="rgb", mask_folder="mask", image_size=(256, 256), transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.rgb_folder = rgb_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        self.transform = transform
        # Define a basic transform if none provided (resize and ToTensor)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
        # For mask, we will only resize and convert to tensor (later threshold it)
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.rgb_folder, os.path.basename(self.data_frame.iloc[idx]['path']))
        mask_name = os.path.join(self.root_dir, self.mask_folder, os.path.basename(self.data_frame.iloc[idx]['path']))

        # Open images
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')

        # Apply transforms
        image = self.transform(image)
        mask = self.mask_transform(mask)
        # For binary segmentation, threshold the mask
        mask = (mask > 0.5).float()

        return image, mask
