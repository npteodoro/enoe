import yaml
import os
import torch

class ConfigLoader:
    def __init__(self, config_path=None, job="", step=""):
        """Initialize the ConfigLoader with the path to the configuration file."""
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = config_path or os.path.join(self.root_dir, "configs", "config.yaml")
        self.job = job
        self.step = step

        self.load()

        model_config = self.config.get("model", {})
        self.encoder_name = model_config.get("encoder_name", None)
        self.log_dir = self.config.get("log_dir", "logs")

        self.device = torch.device(self.config.get("device", "cuda") \
                                    if torch.cuda.is_available() else "cpu")

    def load(self):
        """Load and return the configuration from a YAML file."""

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if self.step:
            # Check if the step exists in the config
            if self.step not in self.config:
                raise ValueError(f"Configuration for step '{self.step}' not found in {self.config_path}.")
            self.config = self.config[self.step]

    def get_root_dir(self):
        """Return the root directory."""
        return self.root_dir

    def get_job(self):
        """Return the job name."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.job

    def get_step(self):
        """Return the step name."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.step

    def get_config(self):
        """Return the loaded configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.config

    def get_device(self):
        """Return the device (CPU or GPU)."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.device

    def get_log_dir(self):
        """Return the log directory."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.log_dir

    def get_model(self):
        """Return the model configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.config.get("model", {})

    def get_encoder_name(self):
        """Return the encoder name."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.encoder_name

    def get_dataset(self):
        """Return the dataset configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.config.get("dataset", {})

    def get_training(self):
        """Return the training configuration."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")
        return self.config.get("training", {})

    def get_model_config(self):
        """Extract model-related configuration details."""
        if not self.config:
            raise ValueError("Configuration not loaded. Call `load()` first.")

        model_config = self.config.get("model", {})
        encoder_name = model_config.get("encoder_name", "timm-mobilenetv3_small_100")
        encoder_weights = model_config.get("encoder_weights", "imagenet")
        in_channels = model_config.get("in_channels", 3)
        classes = model_config.get("classes", 1)
        return encoder_name, encoder_weights, in_channels, classes



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
