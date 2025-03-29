# architectures/dual_input_model.py
import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, mobilenet_v3_small, mobilenet_v3_large

def get_backbone(backbone_name, channels=3, pretrained=True):
    """Factory function to get the right backbone with correct input channels"""
    if backbone_name == "shufflenet":
        model = shufflenet_v2_x0_5(pretrained=pretrained)
        if channels == 1:  # For mask input
            model.conv1[0] = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False)
        return model, model.fc.in_features
        
    elif backbone_name == "mobilenet_small":
        model = mobilenet_v3_small(pretrained=pretrained)
        if channels == 1:  # For mask input
            model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return model, model.classifier[0].in_features
        
    elif backbone_name == "mobilenet_large":
        model = mobilenet_v3_large(pretrained=pretrained)
        if channels == 1:  # For mask input
            model.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        return model, model.classifier[0].in_features
        
    # Add more options here as needed
    
    else:
        raise ValueError(f"Backbone {backbone_name} not supported")

def get_dual_input_model(backbone_name="shufflenet", num_classes=4, pretrained=True):
    """
    Creates a dual-input model with specified backbone
    backbone_name: "shufflenet" or "mobilenet"
    """
    # RGB branch
    rgb_model, rgb_features = get_backbone(backbone_name, channels=3, pretrained=pretrained)
    
    # Mask branch
    mask_model, mask_features = get_backbone(backbone_name, channels=1, pretrained=False)
    
    # Create a model that combines both features
    class DualInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Get all layers except the final classifier
            self.rgb_branch = nn.Sequential(*list(rgb_model.children())[:-1])
            self.mask_branch = nn.Sequential(*list(mask_model.children())[:-1])
            
            # Add this line to define the pool attribute
            self.pool = nn.AdaptiveAvgPool2d(1)
            
            # Create classifier for the combined features
            self.classifier = nn.Sequential(
                nn.Linear(rgb_features + mask_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes)
            )
            
        def forward(self, rgb, mask):
            # Add proper pooling
            rgb_features = self.pool(self.rgb_branch(rgb)).flatten(1)
            mask_features = self.pool(self.mask_branch(mask)).flatten(1)
            combined = torch.cat([rgb_features, mask_features], dim=1)
            return self.classifier(combined)
            
    return DualInputModel()