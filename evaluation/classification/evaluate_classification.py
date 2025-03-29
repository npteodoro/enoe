# evaluation/evaluate_classification.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from architectures.classification.dual_input_classifier import get_dual_input_model
from data.loaders.classification_loader import ClassificationDataset, get_classification_dataloader
from utils.logger import get_logger, log_config, log_model_info
from utils.config import load_config

def main():
    # Load configuration from YAML file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = load_config(os.path.join(project_root, "configs/config_classification.yaml"))

    device = torch.device(config.get("device", "cpu"))
    
    # Extract model configuration details
    model_config = config["model"]
    encoder_name = model_config.get("encoder_name", "mobilenetv3_small_classifier")
    num_classes = model_config.get("num_classes", 4)
    use_mask = model_config.get("use_mask", False)  # Add this line
    
    # Setup TensorBoard logger for evaluation in a dedicated subfolder
    eval_log_dir = os.path.join(config.get("log_dir", "logs"), "evaluation", "classification", encoder_name)
    os.makedirs(eval_log_dir, exist_ok=True)
    writer = get_logger(eval_log_dir)
    
    # Log configuration and model info
    log_config(writer, config)
    log_model_info(writer, encoder_name)
    
    # Create dataset and dataloader
    dataset_config = config["dataset"]
    transform = transforms.Compose([
        transforms.Resize(tuple(dataset_config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataloader = get_classification_dataloader(
        csv_file=dataset_config["csv_file"],
        root_dir=dataset_config["root_dir"],
        rgb_folder=dataset_config.get("rgb_folder", "rgb"),
        mask_folder=dataset_config.get("mask_folder", "mask") if use_mask else None,
        batch_size=config["training"].get("batch_size", 4),
        shuffle=False,
        num_workers=config["training"].get("num_workers", 2),
        transform=transform
    )
    
    # Initialize classification model
    backbone_name = model_config.get("backbone_name", "shufflenet")
    model = get_dual_input_model(backbone_name=backbone_name, num_classes=num_classes, pretrained=False)
    
    # Construct model checkpoint path (assumes the model was saved under models/classification/<encoder_name>/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(project_root, "models", "classification", encoder_name, f"{encoder_name}.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the classifier with encoder '{encoder_name}' first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if use_mask:
                images, masks, labels = batch
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(images, masks)
            else:
                images, _, labels = batch  # Ignore mask
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    print(f"Classification Accuracy: {accuracy:.4f}")
    writer.add_scalar("Eval/Accuracy", accuracy)
    writer.close()

if __name__ == "__main__":
    main()
