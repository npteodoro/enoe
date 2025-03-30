import torch
import torch.nn as nn

class NanoSingle(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2),  # More channels vs dual
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.temp = nn.Sequential(
            nn.Conv1d(12+1, 16, 3),  # 12(rgb)+1(time)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*3, num_classes)
        )

    def forward(self, x, t):
        batch, seq = x.shape[:2]
        rgb_feats = self.rgb(x.view(-1,3,128,128)).view(batch, seq, -1)
        combined = torch.cat([rgb_feats, t.unsqueeze(-1)], -1)
        return self.temp(combined.permute(0,2,1))