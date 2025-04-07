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
        self.encoder_name = model_config.get("encoder_name", "none")
        self.log_dir = self.config.get("log_dir", "logs")

    def require_config(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'config') or self.config is None:
                raise ValueError("Configuration not loaded. Call `load()` first.")
            return func(self, *args, **kwargs)
        return wrapper

    @require_config
    def get_root_dir(self):
        """Return the root directory."""
        return self.root_dir

    @require_config
    def get_job(self):
        """Return the job name."""
        return self.job

    @require_config
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

    @require_config
    def get_log_dir(self):
        """Return the log directory."""
        return self.log_dir

    def set_encoder_name(self, encoder_name):
        """Set the encoder name."""
        self.encoder_name = encoder_name

    def set_model_architecture(self, architecture):
        """Set the architecture."""
        self.config["model"]["architecture"] = architecture

    def set_parameter(self, key_path, value):
        """
        Set a parameter in the configuration using a key path.
        Example: key_path="dataset.csv_file", value="file.csv"
        """
        print(f"Setting parameter '{key_path}' to '{value}'")
        keys = key_path.split(".")
        config_section = self.config

        # Traverse the nested dictionary to the last key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}  # Create nested dictionaries if they don't exist
            config_section = config_section[key]

        # Attempt to convert the value to float, int, or leave as string
        final_key = keys[-1]
        try:
            # Try to convert to float
            value = float(value)
            # If it can be converted to float but is an integer (e.g., 5.0), convert to int
            if value.is_integer():
                value = int(value)
        except ValueError:
            # If conversion to float fails, leave as string
            pass

        # Set the value at the final key
        config_section[final_key] = value
