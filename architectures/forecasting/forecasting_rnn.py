# architectures/forecasting/forecasting_cnn_gru.py
import torch
import torch.nn as nn
import torchvision.models as models

class ForecastingCNN_GRU(nn.Module):
    def __init__(self, time_window=7, num_classes=4, cnn_output_size=256,
                 gru_hidden_size=128, gru_num_layers=2):  # Added gru_num_layers parameter
        super().__init__()

        # CNN feature extractor (MobileNetV3 Small)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # Remove the classifier
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])

        # Add flatten layer
        self.flatten = nn.Flatten()

        # Get the correct number of output features
        self.fc = nn.Linear(mobilenet.classifier[0].in_features, cnn_output_size)

        # RNN for sequence processing
        self.gru = nn.GRU(
            input_size=cnn_output_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,  # Now using the parameter instead of hardcoded value
            batch_first=True,
            dropout=0.2 if gru_num_layers > 1 else 0  # Only use dropout if multiple layers
        )

        # Final classifier
        self.classifier = nn.Linear(gru_hidden_size, num_classes)

    def forward(self, x):
        # x shape: [B, T, C, H, W]
        batch_size, seq_len, C, H, W = x.size()
        features = []
        # Process each time step separately
        for t in range(seq_len):
            xt = x[:, t]  # [B, C, H, W]
            feat = self.feature_extractor(xt)
            feat = self.flatten(feat)  # Now using the defined flatten layer
            feat = self.fc(feat)
            features.append(feat)
        # Stack features along time dimension: [B, T, cnn_output_size]
        features = torch.stack(features, dim=1)
        # Process sequence with GRU
        gru_out, _ = self.gru(features)
        # Use the output at the last time step for forecasting
        last_out = gru_out[:, -1, :]
        output = self.classifier(last_out)
        return output
