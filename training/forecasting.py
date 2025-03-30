import torch.nn as nn
import torch.optim as optim

from data.loaders.forecasting import get_forecasting_dataloader
from architectures.forecasting.forecasting_rnn import ForecastingCNN_GRU
from jobs.training import TrainingStep

class TrainingForecasting(TrainingStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

    def define_dataloader(self):
        """
        Define the dataloader configuration.
        """
        self.dataloader = get_forecasting_dataloader(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            rgb_folder=self.config_dataset.get("rgb_folder"),
            batch_size=self.data_training.get("batch_size"),
            time_window=self.config_dataset.get("time_window"),
            num_workers=self.data_training.get("num_workers")
        )

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        self.model = ForecastingCNN_GRU(
            cnn_output_size=self.config_model.get("cnn_output_size"),
            gru_hidden_size=self.config_model.get("gru_hidden_size"),
            gru_num_layers=self.config_model.get("gru_num_layers"),
        ).to(self.device)

    def run_model(self):
        """
        Run the model training.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config_training.get("learning_rate"))

        num_epochs = self.config_training.get("num_epochs")
        total_samples = len(self.dataloader.dataset)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for imgs, target in self.dataloader:
                imgs, target = imgs.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(imgs)  # outputs shape: [B, 1]
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                self.logger.add_scalar("Train/BatchLoss", loss.item(), global_step)
                global_step += 1

            epoch_loss = running_loss / total_samples
            self.logger.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
