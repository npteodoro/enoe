# utils/config.py
import yaml

def load_config(config_path):
    """Load and return the configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_model_config(config):
    """Extract model-related configuration details."""
    model_config = config.get("model", {})
    encoder_name = model_config.get("encoder_name", "timm-mobilenetv3_small_100")
    encoder_weights = model_config.get("encoder_weights", "imagenet")
    in_channels = model_config.get("in_channels", 3)
    classes = model_config.get("classes", 1)
    return encoder_name, encoder_weights, in_channels, classes

