from abc import ABC, abstractmethod
import torch

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

    @abstractmethod
    def process(self):
        pass
