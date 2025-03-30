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

    def define_transform(self):
        """
        Define the transformations for the dataset.
        """
        self.transform = None

    def define_dataset(self):
        """
        Define the dataset configuration.
        """
        self.dataset = None

    def define_dataloader(self):
        """
        Define the dataloader configuration.
        """
        self.dataloader = None

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        self.model = None

    def run_model(self):
        """
        Run the model on a sample batch.
        """
        pass
