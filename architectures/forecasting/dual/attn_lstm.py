import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnLSTMDual(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # RGB Encoder
        self.rgb_enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Mask Encoder
        self.mask_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Attention LSTM
        self.lstm = nn.LSTM(128+64+1, 256, batch_first=True)  # +time
        self.attn = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias=False)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, masks, t):
        batch, seq = x.shape[:2]
        rgb_feats = self.rgb_enc(x.view(-1,3,128,128)).view(batch, seq, -1)
        mask_feats = self.mask_enc(masks.view(-1,1,128,128)).view(batch, seq, -1)
        combined = torch.cat([rgb_feats, mask_feats, t.unsqueeze(-1)], -1)
        
        lstm_out, _ = self.lstm(combined)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(context)