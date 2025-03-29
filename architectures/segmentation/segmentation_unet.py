# architectures/unet_mobilenet_v3.py
import segmentation_models_pytorch as smp

def get_unet_mobilenet_v3(in_channels=3,
                          classes=1,
                          encoder_name="timm-mobilenetv3_small_100",
                          encoder_weights="imagenet"):
    """
    Returns a UNet model with a MobileNetV3 encoder.

    Args:
        in_channels (int): Number of input channels.
        classes (int): Number of output segmentation classes.
        encoder_weights (str): Pretrained weights for the encoder.

    Returns:
        model: An instance of smp.Unet.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model
