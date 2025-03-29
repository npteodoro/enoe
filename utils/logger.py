# utils/logger.py
from torch.utils.tensorboard import SummaryWriter
import os

def get_logger(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer

