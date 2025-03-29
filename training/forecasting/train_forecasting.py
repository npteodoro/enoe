# training/train_forecasting.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data.loaders.forecasting_loader import get_forecasting_dataloader
from architectures.forecasting.forecasting_rnn import ForecastingCNN_GRU
from utils.logger import get_logger, log_config
from utils.config import load_config
import torchvision.transforms as transforms

def main():
    config = load_config("configs/config_forecasting.yaml")
    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # Setup TensorBoard logger
    log_dir = config["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = get_logger(log_dir)
    log_config(writer, config)

    # Create forecasting dataloader
    dataset_config = config["dataset"]
    dataloader = get_forecasting_dataloader(
        csv_file=dataset_config["csv_file"],
        rgb_folder=dataset_config["rgb_folder"],
        batch_size=config["training"]["batch_size"],
        time_window=dataset_config["time_window"],
        num_workers=config["training"]["num_workers"]
    )

    # Initialize forecasting model
    model_config = config["model"]
    model = ForecastingCNN_GRU(
        cnn_output_size=model_config["cnn_output_size"],
        gru_hidden_size=model_config["gru_hidden_size"],
        gru_num_layers=model_config["gru_num_layers"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    total_samples = len(dataloader.dataset)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # outputs shape: [B, 1]
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / total_samples
        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save the forecasting model checkpoint
    model_dir = "models/forecasting"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "forecasting_cnn_gru.pth"))
    writer.close()
    print("Forecasting model training complete and saved.")

if __name__ == "__main__":
    main()
