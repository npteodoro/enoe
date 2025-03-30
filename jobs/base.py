from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
import importlib

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

        print(f"Encoder name: {self.encoder_name}")

        self.device = torch.device(self.config.get_config().get("device", "cuda") \
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

    def define_dataloader(self, batch_size=None, num_workers=None, shuffle=False):
        """
        Define the dataloader configuration.
        """
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size or self.config_training.get("batch_size", 4),
            num_workers=num_workers or self.config_training.get("num_workers", 2),
            shuffle=shuffle
        )

    def initialize_model(self):
        """
        Dynamically initialize the model based on the class name provided in the configuration.
        """
        architecture = f"architectures.{self.config.get_step()}.{self.config_model.get("architecture")}"
        print(f"Model architecture: {architecture}")

        # Dynamically import the model class
        module_name, class_name = architecture.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import model class '{architecture}': {e}")

        # Initialize the model
        self.model = model_class().to(self.device)

    def run_model(self):
        """
        Run the model on a sample batch.
        """
        pass
