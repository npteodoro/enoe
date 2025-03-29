# utils/config.py
import yaml
import os

def load_config(job="", step="", config_path=None):
    """Load and return the configuration from a YAML file."""

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if config_path is None:
        config_path = os.path.join(root_dir, "configs/config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if step:
        # Check if the step exists in the config
        if step not in config:
            raise ValueError(f"Configuration for step '{step}' not found in {config_path}.")
        config = config[step]

    # Register root_dir
    config["root_dir"] = root_dir
    config["job"] = job
    config["step"] = step

    return config

def get_model_config(config):
    """Extract model-related configuration details."""
    model_config = config.get("model", {})
    encoder_name = model_config.get("encoder_name", "timm-mobilenetv3_small_100")
    encoder_weights = model_config.get("encoder_weights", "imagenet")
    in_channels = model_config.get("in_channels", 3)
    classes = model_config.get("classes", 1)
    return encoder_name, encoder_weights, in_channels, classes
