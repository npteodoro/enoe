import os
import torch
from jobs.base import Job
from jobs.base import Step

class Evaluation(Job):
    def execute(self, step):
        step.process()

class EvaluationStep(Step):
    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)

    def load_model(self):
        """
        Load the model weights from the specified path.
        """
        model_path = self.config.get_model_path()
        print(f"Loading model from {model_path}...")

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print(f"Please train the model with encoder '{self.encoder_name}' first.")
            exit(-1)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self):
        pass

    def process(self):

        self.define_transform()

        self.define_dataset()

        self.define_dataloader()

        self.define_model_parameters()

        self.initialize_model()

        self.load_model()

        self.evaluate()
