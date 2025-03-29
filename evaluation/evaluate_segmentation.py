# evaluation/evaluate_segmentation.py
import os
import torch
import yaml
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

    # Load model configuration
    model_config = config["model"]
    
    # Initialize model with same parameters as training
    model = get_unet_mobilenet_v3(
        in_channels=model_config["in_channels"],
        classes=model_config["classes"],
        encoder_name=model_config["encoder_name"],
        encoder_weights=model_config["encoder_weights"]
    )
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Use absolute paths for more reliability
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             "models", f"{model_config['encoder_name']}.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Make sure you've trained the model with encoder '{model_config['encoder_name']}' first")
        return
        
    # Load weights with proper path
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Setup logger for evaluation
    writer = get_logger("logs/evaluation")

    # Create dataloader from config
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
        batch_size=config["training"].get("batch_size", 4), 
        shuffle=False, 
        num_workers=config["training"].get("num_workers", 2)
    )

    # Rest of your evaluation code
    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = torch.sigmoid(model(images))
            batch_iou = iou_score(outputs, masks)
            batch_dice = dice_loss(outputs, masks)
            total_iou += batch_iou
            total_dice += batch_dice
            num_batches += 1
            # Log per-batch evaluation metrics
            writer.add_scalar("Eval/Batch_IoU", batch_iou, batch_idx)
            writer.add_scalar("Eval/Batch_DiceLoss", batch_dice, batch_idx)

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    print(f"Average IoU: {avg_iou:.4f}, Average Dice Loss: {avg_dice:.4f}")
    writer.add_scalar("Eval/Avg_IoU", avg_iou)
    writer.add_scalar("Eval/Avg_DiceLoss", avg_dice)
    writer.close()

if __name__ == "__main__":
    main()

