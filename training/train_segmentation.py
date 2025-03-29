# training/train_segmentation.py
import os
import yaml
import torch
from torch.utils.data import DataLoader
from architectures.unet_mobilenet_v3 import get_unet_mobilenet_v3
from data.loaders import RiverSegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss
from utils.logger import get_logger
import torchvision.transforms as transforms

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config("configs/config_segmentation.yaml")

    device = torch.device(config.get("device", "cpu"))

    # Setup TensorBoard logger
    writer = get_logger(config.get("log_dir", "logs"))

    # Create dataset and dataloader
    dataset_config = config["dataset"]
    transform = transforms.Compose([
        transforms.Resize(tuple(dataset_config["image_size"])),
        transforms.ToTensor(),
        # You can add normalization here if required
    ])

    dataset = RiverSegmentationDataset(
        csv_file=dataset_config["csv_file"],
        root_dir=dataset_config["root_dir"],
        rgb_folder=dataset_config.get("rgb_folder", "rgb"),
        mask_folder=dataset_config.get("mask_folder", "mask"),
        image_size=tuple(dataset_config["image_size"]),
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    # Initialize model
    model_config = config["model"]
    model = get_unet_mobilenet_v3(
        in_channels=model_config["in_channels"],
        classes=model_config["classes"],
        encoder_name=model_config["encoder_name"],
        encoder_weights=model_config["encoder_weights"]
    )
    model = model.to(device)

    # Loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    total_samples = len(dataset)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [B, 1, H, W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # Log loss at each batch
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / total_samples

        # Evaluate IoU and Dice on a small validation batch (here we reuse training data for simplicity)
        model.eval()
        with torch.no_grad():
            sample_images, sample_masks = next(iter(dataloader))
            sample_images = sample_images.to(device)
            sample_masks = sample_masks.to(device)
            sample_outputs = torch.sigmoid(model(sample_images))
            iou = iou_score(sample_outputs, sample_masks)
            dice = dice_loss(sample_outputs, sample_masks)

        # Log epoch-level metrics
        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Train/IoU", iou, epoch)
        writer.add_scalar("Train/DiceLoss", dice, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, IoU: {iou:.4f}, Dice Loss: {dice:.4f}")

    # Save model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_config['encoder_name']}.pth")
    writer.close()
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()

