# data/forecasting_loaders.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import torch.nn as nn

class ForecastingDataset(Dataset):
    def __init__(self, csv_file, root_dir, rgb_folder="rgb", time_window=7, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with columns "path", "timestamp", "level".
            root_dir (str): Directory with all the images.
            rgb_folder (str): Folder containing RGB images.
            time_window (int): Number of consecutive images to use as input.
            transform: Transformations to apply to each image.
        """
        self.data = pd.read_csv(csv_file)
        # Ensure the data is sorted by timestamp
        self.data = self.data.sort_values(by="datetime").reset_index(drop=True)
        self.root_dir = root_dir
        self.rgb_folder = rgb_folder
        self.time_window = time_window
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        # The number of sequences available is total_samples - time_window
        return len(self.data) - self.time_window

    def __getitem__(self, idx):
        images = []
        for i in range(self.time_window):
            row = self.data.iloc[idx + i]
            img_path = os.path.join(self.root_dir, self.rgb_folder, os.path.basename(row["path"]))
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            images.append(image)
        # Stack images to create a sequence: [T, C, H, W]
        sequence = torch.stack(images, dim=0)

        # Target level is from the next record after the sequence
        target_level_str = self.data.iloc[idx + self.time_window]["level"]

        # Map string level to numeric value
        level_map = {'low': 0, 'medium': 1, 'high': 2, 'flood': 3}
        target_level = level_map.get(target_level_str.lower(), 0)  # Default to 0 if not found

        target = torch.tensor(target_level, dtype=torch.long)
        return sequence, target

def get_forecasting_dataloader(csv_file, root_dir, rgb_folder, batch_size=16, time_window=7, num_workers=4):
    dataset = ForecastingDataset(csv_file, root_dir, rgb_folder, time_window=time_window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

class ForecastingModel(nn.Module):
    def __init__(self, time_window=7, num_classes=4, cnn_output_size=256, hidden_size=128):
        super().__init__()

        # CNN feature extractor (MobileNetV3 Small)
        mobilenet = mobilenet_v3_small(pretrained=True)
        # Remove the classifier
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])

        # Get the correct number of output features
        self.fc = nn.Linear(mobilenet.classifier[0].in_features, cnn_output_size)

        # RNN for sequence processing
        self.gru = nn.GRU(
            input_size=cnn_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Final classifier
        self.classifier = nn.Linear(hidden_size, num_classes)
