import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from architectures.classification.classifier_dual_input import get_dual_input_model
from data.loaders.classification_loader import get_classification_dataloader
from utils.logger import get_logger, log_config, log_model_info
from utils.config import load_config
import torchvision.transforms as transforms

def main():
    # Load configuration for classification (assumed to be in config_classification.yaml)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(os.path.join(project_root, "configs/config_classification.yaml"))

    device = torch.device(config.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    # Get model config details
    model_config = config["model"]
    encoder_name = model_config.get("encoder_name", "mobilenetv3_small_classifier")
    num_classes = model_config.get("num_classes", 4)  # e.g., 4 classes: low, medium, high, flood
    use_mask = model_config.get("use_mask", False)  # Add a config option for this

    # Get backbone name from config
    backbone_name = model_config.get("backbone_name", "shufflenet")

    # Setup logger for classification training
    log_dir = os.path.join(config.get("log_dir", "logs/training"), "training", encoder_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = get_logger(log_dir)
    log_config(writer, config)
    log_model_info(writer, encoder_name)

    # Create classification dataloader
    dataset_config = config["dataset"]
    transform = transforms.Compose([
        transforms.Resize(tuple(dataset_config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataloader = get_classification_dataloader(
        csv_file=dataset_config["csv_file"],
        rgb_folder=dataset_config["rgb_folder"],
        mask_folder=dataset_config["mask_folder"],
        root_dir=dataset_config["root_dir"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        transform=transform
    )

    # Initialize classification model
    model = get_dual_input_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=True)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    num_epochs = config["training"]["num_epochs"]
    total_samples = len(dataloader.dataset)
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0

        # Current code has incorrect unpacking - should handle masks too
        if use_mask:
            for (images, masks, labels) in dataloader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, masks)  # Pass both images and masks
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()

                writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
                global_step += 1
        else:
            for (images, _, labels) in dataloader:  # Ignore masks
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()

                writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
                global_step += 1

        epoch_loss = running_loss / total_samples
        epoch_acc = correct / total_samples
        writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Train/EpochAccuracy", epoch_acc, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Save classifier checkpoint
    model_dir = os.path.join(project_root, "models", "classification", encoder_name)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, f"{encoder_name}.pth"))
    writer.close()
    print("Classification training complete and model saved.")

if __name__ == "__main__":
    main()
