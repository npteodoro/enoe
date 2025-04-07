import os
import torch

from jobs.base import Job
from jobs.base import Step

class Training(Job):
    def execute(self, step):
        print(f"[Training] Executing {step.__class__.__name__}")
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
        model_dir = os.path.join(self.config.get_root_dir(), "models",
                                 self.config.get_step(), self.encoder_name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"{self.encoder_name}.pth"))
        print("Training complete and model saved.")

    def define_loss(self):
        """
        Define the loss function.
        """
        pass

    def define_optimizer(self):
        """
        Define the optimizer.
        """
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config_training.get("learning_rate")
        )

    def process(self):

        self.define_transform()

        self.define_dataset()

        self.define_dataloader()

        self.initialize_model()

        self.define_loss()

        self.define_optimizer()

        self.run_model()

        self.save_model()
