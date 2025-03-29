# utils/logger.py
from torch.utils.tensorboard import SummaryWriter
import os

def get_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def log_config(writer, config):
    """Logs the configuration dictionary as text."""
    import json
    config_text = json.dumps(config, indent=4)
    writer.add_text("Config", config_text)

def log_model_info(writer, encoder_name):
    """Logs the model name or encoder information."""
    writer.add_text("Model Info", f"Using encoder: {encoder_name}")

