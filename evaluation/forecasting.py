# evaluation/evaluate_forecasting.py
import os
import torch
from torch.utils.data import DataLoader
from data.loaders.forecasting import ForecastingDataset
from architectures.forecasting.forecasting_rnn import ForecastingCNN_GRU
from utils.logger import init_logger
from utils.config import load_config
import torchvision.transforms as transforms

def main(config=None, writer=None, device=None):

    # Create forecasting dataset and dataloader
    dataset_config = config["dataset"]
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = ForecastingDataset(
        csv_file=dataset_config["csv_file"],
        root_dir=dataset_config["root_dir"],
        rgb_folder=dataset_config["rgb_folder"],
        time_window=dataset_config["time_window"],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=config["training"].get("batch_size", 16),
                              shuffle=False, num_workers=config["training"].get("num_workers", 4))

    # Initialize forecasting model
    model_config = config["model"]
    model = ForecastingCNN_GRU(
        cnn_output_size=model_config["cnn_output_size"],
        gru_hidden_size=model_config["gru_hidden_size"],
        gru_num_layers=model_config["gru_num_layers"],
        output_size=model_config["output_size"]
    ).to(device)

    # Construct model checkpoint path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "forecasting", "forecasting_cnn_gru.pth")
    if not os.path.exists(model_path):
        print(f"Error: Forecasting model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_error = 0.0
    total_samples = 0
    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device)
            outputs = model(imgs)
            batch_error = torch.abs(outputs.squeeze() - target).sum().item()
            total_error += batch_error
            total_samples += imgs.size(0)
    avg_error = total_error / total_samples
    print(f"Average Forecasting MAE: {avg_error:.4f}")
    writer.add_scalar("Eval/MAE", avg_error)

if __name__ == "__main__":
    # Load configuration using our config helper
    config = load_config(job="evaluantion", step="forecasting")

    # Setup TensorBoard logger for evaluation in a dedicated subfolder
    writer = init_logger(config=config)

    # Default is cuda if available, else cpu
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    main(config=config, writer=writer, device=device)

    writer.close()
