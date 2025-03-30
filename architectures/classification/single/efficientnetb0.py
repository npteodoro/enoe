from efficientnet_pytorch import EfficientNet
import torch.nn as nn

class SingleEfficientNetB0(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        features = self.model.extract_features(x).mean([2, 3])
        return self.classifier(features)
