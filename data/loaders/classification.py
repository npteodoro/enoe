import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, rgb_folder="rgb", mask_folder="mask", image_size=(224, 224), transform=None):
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        # For mask, we will only resize and convert to tensor (later threshold it)
        if self.mask_folder:
            self.mask_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.rgb_folder, \
                                os.path.basename(self.data_frame.iloc[idx]['path']))
        mask_name = os.path.join(self.root_dir, self.mask_folder, \
                                 os.path.basename(self.data_frame.iloc[idx]['path'])) if self.mask_folder else None

        # Open images
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L") if mask_name else None

        # Convert string labels to integers
        label_str = self.data_frame.iloc[idx]['level']
        label_map = {'low': 0, 'medium': 1, 'high': 2, 'flood': 3}
        label = label_map.get(label_str.lower(), 0)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if mask is not None and self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label
