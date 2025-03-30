import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch

class DualMobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB stream
        self.rgb_stream = mobilenet_v3_small(pretrained=True)
        self.rgb_stream.features[0][0] = nn.Conv2d(3, 16, 3, stride=2, padding=1)

        # Mask stream
        self.mask_stream = mobilenet_v3_small(pretrained=False)
        self.mask_stream.features[0][0] = nn.Conv2d(1, 16, 3, stride=2, padding=1)

        # Feature fusion
        self.classifier = nn.Sequential(
            nn.Linear(576*2, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, mask):
        rgb_feats = self.rgb_stream.features(rgb).mean([2, 3])
        mask_feats = self.mask_stream.features(mask).mean([2, 3])
        return self.classifier(torch.cat([rgb_feats, mask_feats], 1))
