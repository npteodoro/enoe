import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

class ClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, rgb_folder="rgb", mask_folder="mask", image_size=(224, 224), transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.rgb_folder = os.path.join(root_dir, rgb_folder)
        self.mask_folder = os.path.join(root_dir, mask_folder) if mask_folder else None
        self.image_size = image_size
        self.transform = transform     
        # Define default transform if not provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        if self.mask_folder:
            self.mask_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.rgb_folder, os.path.basename(self.data_frame.iloc[idx]['path']))
        mask_name = os.path.join(self.mask_folder, os.path.basename(self.data_frame.iloc[idx]['path'])) if self.mask_folder else None
        
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L") if mask_name else None
        
        # Convert string labels to integers
        label_str = self.data_frame.iloc[idx]['level']
        label_map = {'low': 0, 'medium': 1, 'high': 2, 'flood': 3}
        label = label_map.get(label_str.lower(), 0)

        if self.transform:
            image = self.transform(image)
        
        if mask is not None and self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = rgb_model.features
        self.mask_branch = mask_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(rgb_features + mask_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes)
        )

def get_classification_dataloader(csv_file, root_dir, rgb_folder="rgb", mask_folder="mask", batch_size=32, shuffle=True, num_workers=4, transform=None):
    dataset = ClassificationDataset(csv_file, root_dir, rgb_folder, mask_folder, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

