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
            self.config.get_job(),
            self.config.get_step(),
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

    def add_scalar(self, tag, value, step):
        """
        Adds a scalar value to the TensorBoard log.
        """
        self.writer.add_scalar(tag, value, step)

    def add_text(self, tag, text, step):
        """
        Adds a text entry to the TensorBoard log.
        """
        self.writer.add_text(tag, text, step)

def log_config(writer, config):
    """Logs the configuration dictionary as text."""

    writer.add_text("Config", json.dumps(config, indent=4))

def log_model_info(writer, encoder_name):
    """Logs the model name or encoder information."""
    writer.add_text("Model Info", f"Using encoder: {encoder_name}")

def init_logger(config=None, log_dir="logs"):

    # Define log dir
    model_config = config.get("model", {})
    encoder_name = model_config.get("encoder_name", "timm-mobilenetv3_small_100")
    log_dir = os.path.join(config.get("log_dir", "logs"), config["job"], config["step"], encoder_name)

    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Log the configuration
    log_config(writer, config)

    # Log the model name or encoder information
    log_model_info(writer, encoder_name)

    return writer
