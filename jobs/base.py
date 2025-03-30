from abc import ABC, abstractmethod
import torch
import os

class Job(ABC):
    @abstractmethod
    def execute(self, step):
        pass

class Step(ABC):

    def __init__(self, config, logger):
        """
        Initialize the Step with configuration and logger.
        """
        self.config = config
        self.logger = logger

        self.config_dataset = config.get_dataset()
        self.config_training = config.get_training()
        self.config_model = config.get_model()

        self.encoder_name = config.get_encoder_name()

        self.device = torch.device(self.config.get("device", "cuda") \
                                    if torch.cuda.is_available() else "cpu")

    def save_model(self):
        """
        Save model checkpoint in a subfolder for the current encoder
        """
        model_dir = os.path.join(self.config.get_root_dir(), "models",
                                 self.config.get_step(), self.encoder_name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"{self.encoder_name}.pth"))
        print("Training complete and model saved.")

    @abstractmethod
    def process(self):
        pass
