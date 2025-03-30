import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class ClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, rgb_folder="rgb", mask_folder="mask", 
                 image_size=(224, 224), use_mask=True, augment=False):
        """
        Improved Features:
        - Automatic label mapping
        - Mask validation and processing
        - Robust error handling
        - Augmentation support
        - Class balancing weights
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.use_mask = use_mask
        self.image_size = image_size
        
        # Validate paths
        self._validate_paths(rgb_folder, mask_folder)
        
        # Create label mapping
        self.label_map, self.class_weights = self._create_label_mapping()
        
        # Configure transforms
        self.transform, self.mask_transform = self._create_transforms(augment)
        
        # Pre-calculate valid samples
        self.valid_indices = self._validate_samples(rgb_folder, mask_folder)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        row = self.data.iloc[real_idx]
        
        try:
            # Load RGB image
            img_path = os.path.join(self.root_dir, 'rgb', os.path.basename(row['path']))
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Load and process mask
            mask = torch.zeros(1, *self.image_size)  # Default dummy mask
            if self.use_mask:
                mask_path = os.path.join(self.root_dir, 'mask', 
                                       os.path.basename(row['path']))
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask)
                mask = (mask > 0.5).float()  # Binarize

            # Process label
            label_str = str(row['level']).lower()
            label = self.label_map.get(label_str, 0)

            return image, mask, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading sample {real_idx}: {str(e)}")
            return None  # Should be filtered by DataLoader

    def _create_transforms(self, augment):
        # Image transforms
        img_transforms = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if augment:
            img_transforms.insert(0, transforms.RandomHorizontalFlip())
            img_transforms.insert(0, transforms.RandomRotation(10))

        # Mask transforms (no normalization)
        mask_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        return transforms.Compose(img_transforms), mask_transforms

    def _create_label_mapping(self):
        unique_labels = self.data['level'].str.lower().unique()
        label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Calculate class weights
        class_counts = self.data['level'].value_counts().sort_index()
        weights = 1. / torch.sqrt(torch.tensor(class_counts.values, dtype=torch.float))
        return label_map, weights

    def _validate_paths(self, rgb_folder, mask_folder):
        # Check RGB folder exists
        if not os.path.exists(os.path.join(self.root_dir, rgb_folder)):
            raise ValueError(f"RGB folder {rgb_folder} not found in {self.root_dir}")
            
        # Check mask folder if required
        if self.use_mask and not os.path.exists(os.path.join(self.root_dir, mask_folder)):
            raise ValueError(f"Mask folder {mask_folder} required but not found")

    def _validate_samples(self, rgb_folder, mask_folder):
        valid_indices = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            try:
                img_path = os.path.join(self.root_dir, rgb_folder, os.path.basename(row['path']))
                if not os.path.exists(img_path):
                    continue
                    
                if self.use_mask:
                    mask_path = os.path.join(self.root_dir, mask_folder, 
                                           os.path.basename(row['path']))
                    if not os.path.exists(mask_path):
                        continue
                        
                valid_indices.append(idx)
            except:
                continue
        return valid_indices

    def get_class_weights(self):
        return self.class_weights