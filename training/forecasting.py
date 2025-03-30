import torch.nn as nn
import torch.optim as optim

from data.loaders.forecasting import ForecastingDataset
from architectures.forecasting.dual.attn_lstm import AttnLSTMDual
from jobs.training import TrainingStep

class TrainingForecasting(TrainingStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

    def define_dataset(self):
        self.dataset = ForecastingDataset(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            time_window=self.config_dataset.get("time_window"),
            transform=self.transform
        )

    def define_dataloader(self):
        super().define_dataloader(shuffle=True)

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        self.model = AttnLSTMDual().to(self.device)

    def run_model(self):
        """
        Run the model training.
        """
        num_epochs = self.config_training.get("num_epochs")
        total_samples = len(self.dataloader.dataset)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, target in self.dataloader:
                imgs, target = imgs.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)  # outputs shape: [B, 1]
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                self.logger.add_scalar("Train/BatchLoss", loss.item(), global_step)
                global_step += 1

            epoch_loss = running_loss / total_samples
            self.logger.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
