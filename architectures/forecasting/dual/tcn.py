import torch
import torch.nn as nn

class TCNDual(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mask_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.tcn = nn.Sequential(
            nn.Conv1d(64+32+1, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, masks, t):
        rgb_feats = self.rgb_enc(x.flatten(0,1)).view(*x.shape[:2], -1)
        mask_feats = self.mask_enc(masks.flatten(0,1)).view(*masks.shape[:2], -1)
        combined = torch.cat([rgb_feats, mask_feats, t.unsqueeze(-1)], -1)
        return self.fc(self.tcn(combined.permute(0,2,1)).squeeze())