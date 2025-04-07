from torch.utils.tensorboard import SummaryWriter
import os
import json

class Logger:
    def __init__(self, config, log_dir="logs"):
        """
        Initialize the Logger with configuration and log directory.
        """
        self.config = config

        self.log_dir = os.path.join(
            self.config.get_log_dir(),
            self.config.get_step(),
            self.config.get_job(),
            self.config.get_encoder_name()
        )

        # Create the log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.log_config()
        self.log_model_info()

    def log_config(self):
        """
        Logs the configuration dictionary as text.
        """
        self.writer.add_text("Config", json.dumps(self.config.get_config(), indent=4))

    def log_model_info(self):
        """
        Logs the model name or encoder information.
        """
        self.writer.add_text("Model Info", f"Using encoder: {self.config.get_encoder_name()}")

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()

    def add_scalar(self, *args, **kwargs):
        """
        Adds a scalar value to the TensorBoard log.
        """
        self.writer.add_scalar(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        """
        Adds a text entry to the TensorBoard log.
        """
        self.writer.add_text(*args, **kwargs)
