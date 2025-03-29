# training/train_segmentation.py
import os
import torch
from torch.utils.data import DataLoader
from architectures.segmentation.segmentation_unet import get_unet_mobilenet_v3
from data.loaders.segmentation_loader import RiverSegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss
from utils.logger import get_logger, log_config, log_model_info
from utils.config import load_config, get_model_config
import torchvision.transforms as transforms

def main():
    # Load configuration using our config helper
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(os.path.join(project_root, "configs/config_segmentation.yaml"))

    encoder_name, encoder_weights, in_channels, classes = get_model_config(config)

    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # Setup TensorBoard logger with a subfolder for the current encoder
    log_dir = os.path.join(config.get("log_dir", "logs/training"), "training", encoder_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = get_logger(log_dir)

    # Log entire configuration and model info
    log_config(writer, config)
    log_model_info(writer, encoder_name)

    # Create dataset and dataloader
    dataset_config = config["dataset"]
    transform = transforms.Compose([
        transforms.Resize(tuple(dataset_config["image_size"])),
        transforms.ToTensor(),
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
    model = get_unet_mobilenet_v3(
        in_channels=in_channels,
        classes=classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
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
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            global_step += 1

        epoch_loss = running_loss / total_samples

        # Evaluate on a small validation batch (using training data here for simplicity)
        model.eval()
        with torch.no_grad():
            sample_images, sample_masks = next(iter(dataloader))
            sample_images = sample_images.to(device)
            sample_masks = sample_masks.to(device)
            sample_outputs = torch.sigmoid(model(sample_images))
            iou = iou_score(sample_outputs, sample_masks)
            dice = dice_loss(sample_outputs, sample_masks)

        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Train/IoU", iou, epoch)
        writer.add_scalar("Train/DiceLoss", dice, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, IoU: {iou:.4f}, Dice Loss: {dice:.4f}")

    # Save model checkpoint in a subfolder for the current encoder
    model_dir = os.path.join(project_root, "models", "segmentation", encoder_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{encoder_name}.pth"))
    writer.close()
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
