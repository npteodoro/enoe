import segmentation_models_pytorch as smp

class SmpUnetMobileNetV3(smp.Unet):
    """
    Custom UNet model with MobileNetV3 encoder.
    """
    def __init__(self,
                 encoder_name="timm-mobilenetv3_small_100",
                 encoder_weights="imagenet",
                 in_channels=3,
                 classes=1):
        super().__init__(encoder_name=encoder_name,
                         encoder_weights=encoder_weights,
                         in_channels=in_channels,
                         classes=classes)
