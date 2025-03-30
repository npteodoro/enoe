import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnLSTMSingle(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),  # Deeper vs dual
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.lstm = nn.LSTM(256+1, 384, batch_first=True)  # Compensate no mask
        self.attn = nn.Sequential(
            nn.Linear(384, 192),
            nn.Tanh(),
            nn.Linear(192, 1, bias=False)
        )
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, x, t):
        batch, seq = x.shape[:2]
        rgb_feats = self.rgb_enc(x.view(-1,3,128,128)).view(batch, seq, -1)
        combined = torch.cat([rgb_feats, t.unsqueeze(-1)], -1)
        
        lstm_out, _ = self.lstm(combined)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)