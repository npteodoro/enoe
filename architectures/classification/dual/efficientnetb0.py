from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch

class DualEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB stream
        self.rgb_stream = EfficientNet.from_pretrained('efficientnet-b0')

        # Mask stream
        self.mask_stream = EfficientNet.from_name('efficientnet-b0')
        self.mask_stream._conv_stem = nn.Conv2d(1, 32, kernel_size=3,
                                              stride=2, padding=1, bias=False)

        # Feature fusion
        self.classifier = nn.Linear(1280*2, num_classes)

    def forward(self, rgb, mask):
        rgb_feats = self.rgb_stream.extract_features(rgb).mean([2, 3])
        mask_feats = self.mask_stream.extract_features(mask).mean([2, 3])
        return self.classifier(torch.cat([rgb_feats, mask_feats], 1))
