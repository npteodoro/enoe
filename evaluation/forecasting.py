import torch
from data.loaders.forecasting import ForecastingDataset
from architectures.forecasting.dual.attn_lstm import AttnLSTMDual
import torchvision.transforms as transforms

from jobs.evaluation import EvaluationStep

class EvaluationForecasting(EvaluationStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

    def define_transform(self):
        """
        Define the transformations for the dataset."
        """
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def define_dataset(self):
        self.dataset = ForecastingDataset(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            transform=self.transform
        )

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        model = AttnLSTMDual().to(self.device)

    def run_model(self):

        total_error = 0.0
        total_samples = 0
        with torch.no_grad():
            for imgs, target in self.dataloader:
                imgs, target = imgs.to(self.device), target.to(self.device)
                outputs = self.model(imgs)
                batch_error = torch.abs(outputs.squeeze() - target).sum().item()
                total_error += batch_error
                total_samples += imgs.size(0)
        avg_error = total_error / total_samples
        print(f"Average Forecasting MAE: {avg_error:.4f}")
        self.logger.add_scalar("Eval/MAE", avg_error)
