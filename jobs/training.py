import os
import torch

from jobs.base import Job
from jobs.base import Step

class Training(Job):
    def execute(self, step):
        step.process()

class TrainingStep(Step):
    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)


    def save_model(self):
        """
        Save model checkpoint in a subfolder for the current encoder
        """
        model_path = self.config.get_model_path()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f"Training complete and model saved in {model_path}")

    def define_loss(self):
        """
        Define the loss function.
        """
        pass

    def define_optimizer(self):
        """
        Define the optimizer.
        """
        print("Define Optimizer")
        print(f"  learning rate: {self.config_training.get('learning_rate')}")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config_training.get("learning_rate")
        )

    def train(self):
        pass

    def process(self):

        self.define_transform()

        self.define_dataset()

        self.define_dataloader()

        self.define_model_parameters()

        self.initialize_model()

        self.define_loss()

        self.define_optimizer()

        self.run_model()

        self.train()
