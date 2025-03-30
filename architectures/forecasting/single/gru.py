import torch
import torch.nn as nn

class GRUSingle(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2),  # More capacity
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.gru = nn.GRU(48+1, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, t):
        rgb_feats = self.rgb_enc(x.flatten(0,1)).unflatten(0, x.shape[:2])
        combined = torch.cat([rgb_feats, t.unsqueeze(-1)], -1)
        gru_out, _ = self.gru(combined)
        return self.fc(gru_out[:, -1])
