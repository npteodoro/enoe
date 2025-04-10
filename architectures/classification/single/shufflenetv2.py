from torchvision.models import shufflenet_v2_x1_0
import torch.nn as nn

class SingleShuffleNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = shufflenet_v2_x1_0(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.model(x))
