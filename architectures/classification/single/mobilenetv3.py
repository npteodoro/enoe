import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class SingleMobileNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = mobilenet_v3_small(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Linear(256, num_classes))

    def forward(self, x):
        features = self.model.features(x).mean([2, 3])
        return self.classifier(features)
