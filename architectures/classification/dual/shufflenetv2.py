import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

class DualShuffleNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB stream
        self.rgb_stream = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.rgb_stream.conv1[0] = nn.Conv2d(3, 24, 3, stride=2, padding=1)

        # Mask stream
        self.mask_stream = shufflenet_v2_x1_0(weights=None)
        self.mask_stream.conv1[0] = nn.Conv2d(1, 24, 3, stride=2, padding=1)

        # Feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(1024*2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, mask):
        rgb_feats = self.rgb_stream(rgb)
        mask_feats = self.mask_stream(mask)
        print(f"RGB Features Shape: {rgb_feats.shape}")
        print(f"Mask Features Shape: {mask_feats.shape}")
        rgb_feats = torch.flatten(rgb_feats, 1)  # Flatten to [batch_size, 1024]
        mask_feats = torch.flatten(mask_feats, 1)  # Flatten to [batch_size, 1024]
        print(f"RGB Flatten Features Shape: {rgb_feats.shape}")
        print(f"Mask Flatten Features Shape: {mask_feats.shape}")
        concatenated_feats = torch.cat([rgb_feats, mask_feats], 1)
        print(f"Concatenated Features Shape: {concatenated_feats.shape}")
        return self.classifier(concatenated_feats)
