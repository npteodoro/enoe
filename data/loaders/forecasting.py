import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class ForecastingDataset(Dataset):
    def __init__(self, csv_file, root_dir, time_window=7, max_gap=120,
                 transform=None, mask_transform=None, use_mask=True):
        """
        Improved Features:
        - Robust sequence validation
        - Proper time difference normalization
        - Configurable input types
        - Better error handling
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
        self.time_window = time_window
        self.max_gap = max_gap
        self.use_mask = use_mask

        # Preprocessing
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.data = self.data.sort_values('datetime').reset_index(drop=True)
        # self._validate_time_consistency()  # AttributeError: 'ForecastingDataset' object has no attribute '_validate_time_consistency'

        # Sequence creation with strict time constraints
        self.sequences = self._create_valid_sequences()

        # Time normalization parameters
        self.time_mean, self.time_std = self._calculate_time_stats()

        # Transforms
        self.transform = transform or self._default_image_transform()
        self.mask_transform = mask_transform or self._default_mask_transform()

        # Label mapping
        self.label_map = {'low':0, 'medium':1, 'high':2, 'flood':3}

    def __getitem__(self, idx):
        seq_indices = self.sequences[idx]
        sample = {
            'images': [],
            'masks': [],
            'time_diffs': [],
            'target': None
        }

        prev_time = None
        for i, idx in enumerate(seq_indices):
            row = self.data.iloc[idx]

            # Load and transform image
            img_path = os.path.join(self.root_dir, 'rgb', os.path.basename(row['path']))
            image = self.transform(Image.open(img_path).convert('RGB'))
            sample['images'].append(image)

            # Load mask if enabled
            if self.use_mask:
                mask_path = os.path.join(self.root_dir, 'mask',
                                        os.path.basename(row['path']))
                mask = self.mask_transform(Image.open(mask_path).convert('L'))
                sample['masks'].append(mask)

            # Calculate normalized time difference
            curr_time = row['datetime']
            if prev_time is not None:
                delta = (curr_time - prev_time).total_seconds() / 60
                delta = (delta - self.time_mean) / self.time_std
            else:
                delta = 0.0
            sample['time_diffs'].append(delta)
            prev_time = curr_time

        # Convert lists to tensors
        sample['images'] = torch.stack(sample['images'])
        if self.use_mask:
            sample['masks'] = torch.stack(sample['masks'])
        sample['time_diffs'] = torch.tensor(sample['time_diffs'], dtype=torch.float32)

        # Get target (next timestamp after sequence)
        target_idx = min(seq_indices[-1] + 1, len(self.data)-1)
        target = self.data.iloc[target_idx]['level']
        sample['target'] = torch.tensor(self._label_to_index(target), dtype=torch.long)

        return sample

    def _create_valid_sequences(self):
        """Create sequences where all consecutive samples are within max_gap"""
        sequences = []
        current_seq = []

        for idx in range(len(self.data)):
            if len(current_seq) == 0:
                current_seq.append(idx)
                continue

            # Check gap with previous element
            prev_time = self.data.iloc[current_seq[-1]]['datetime']
            curr_time = self.data.iloc[idx]['datetime']
            gap = (curr_time - prev_time).total_seconds() / 60

            if gap > self.max_gap:
                # Finalize current sequence
                if len(current_seq) >= self.time_window:
                    sequences += self._split_sequence(current_seq)
                current_seq = []

            current_seq.append(idx)

            # Split if sequence becomes too long
            if len(current_seq) >= self.time_window * 2:
                sequences += self._split_sequence(current_seq)
                current_seq = current_seq[-self.time_window:]

        # Add remaining sequences
        if len(current_seq) >= self.time_window:
            sequences += self._split_sequence(current_seq)

        return sequences

    def _split_sequence(self, seq):
        return [seq[i:i+self.time_window]
               for i in range(0, len(seq)-self.time_window+1, self.time_window//2)]

    def _calculate_time_stats(self):
        """Calculate mean/std of time gaps for normalization"""
        deltas = []
        prev_time = None
        for _, row in self.data.iterrows():
            if prev_time is not None:
                delta = (row['datetime'] - prev_time).total_seconds() / 60
                deltas.append(delta)
            prev_time = row['datetime']
        return np.mean(deltas), np.std(deltas)

    def _default_image_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _default_mask_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def _label_to_index(self, label):
        return self.label_map.get(str(label).lower(), 0)

    def __len__(self):
        return len(self.sequences)