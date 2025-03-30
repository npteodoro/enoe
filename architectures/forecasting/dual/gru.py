import torch
import torch.nn as nn

class GRUDual(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mask_enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.gru = nn.GRU(32+16+1, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, masks, t):
        rgb_feats = self.rgb_enc(x.flatten(0,1)).unflatten(0, x.shape[:2])
        mask_feats = self.mask_enc(masks.flatten(0,1)).unflatten(0, masks.shape[:2])
        combined = torch.cat([rgb_feats, mask_feats, t.unsqueeze(-1)], -1)
        gru_out, _ = self.gru(combined)
        return self.fc(gru_out[:, -1])