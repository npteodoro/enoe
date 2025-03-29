# utils/logger.py
from torch.utils.tensorboard import SummaryWriter
import os
import json

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
