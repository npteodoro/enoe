import torch
import torch.nn as nn

class NanoDual(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB branch
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, groups=3),  # Depthwise
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Mask branch
        self.mask = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Temporal (1DConv)
        self.temp = nn.Sequential(
            nn.Conv1d(8+4+1, 16, 3),  # 8(rgb)+4(mask)+1(time)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*3, num_classes)  # Adjust for seq_len
        )

    def forward(self, x, masks, t):
        batch, seq = x.shape[:2]
        rgb_feats = self.rgb(x.view(-1,3,128,128)).view(batch, seq, -1)
        mask_feats = self.mask(masks.view(-1,1,128,128)).view(batch, seq, -1)
        combined = torch.cat([rgb_feats, mask_feats, t.unsqueeze(-1)], -1)
        return self.temp(combined.permute(0,2,1))