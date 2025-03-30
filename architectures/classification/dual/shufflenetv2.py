from torchvision.models import shufflenet_v2_x1_0
import torch.nn as nn
import torch

class DualShuffleNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB stream
        self.rgb_stream = shufflenet_v2_x1_0(pretrained=True)
        self.rgb_stream.conv1[0] = nn.Conv2d(3, 24, 3, stride=2, padding=1)

        # Mask stream
        self.mask_stream = shufflenet_v2_x1_0(pretrained=False)
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
        return self.classifier(torch.cat([rgb_feats, mask_feats], 1))
