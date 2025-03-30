import yaml
import os
import functools

class ConfigLoader:
    def __init__(self, config_path=None):
        """Initialize the ConfigLoader with the path to the configuration file."""
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = config_path or os.path.join(self.root_dir, "configs", "config.yaml")

    def load(self, job: str, step: str):
        """Load and return the configuration from a YAML file."""

        self.job = job
        self.step = step

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        if self.step:
            # Check if the step exists in the config
            if self.step not in self.config:
                raise ValueError(f"Configuration for step '{self.step}' not found in {self.config_path}.")
            self.config = self.config[self.step]

        model_config = self.config.get("model", {})
        self.encoder_name = model_config.get("encoder_name", None)
        self.log_dir = self.config.get("log_dir", "logs")

    def require_config(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("Configuration not loaded. Call `load()` first.")
            return func(self, *args, **kwargs)
        return wrapper

    def get_root_dir(self):
        """Return the root directory."""
        return self.root_dir

    def get_job(self):
        """Return the job name."""
        return self.job

    def get_step(self):
        """Return the step name."""
        return self.step

    @require_config
    def get_config(self):
        """Return the loaded configuration."""
        return self.config

    @require_config
    def get_model(self):
        """Return the model configuration."""
        return self.config.get("model", {})

    @require_config
    def get_dataset(self):
        """Return the dataset configuration."""
        return self.config.get("dataset", {})

    @require_config
    def get_training(self):
        """Return the training configuration."""
        return self.config.get("training", {})

    @require_config
    def get_encoder_name(self):
        """Return the encoder name."""
        return self.encoder_name

    def set_encoder_name(self, encoder_name):
        """Set the encoder name."""
        self.encoder_name = encoder_name

    @require_config
    def get_log_dir(self):
        """Return the log directory."""
        return self.log_dir

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
