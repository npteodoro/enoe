# evaluation/evaluate_segmentation.py
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from architectures.segmentation.segmentation_unet import get_unet_mobilenet_v3
from data.loaders.segmentation import RiverSegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss
from utils.logger import init_logger
from utils.config import load_config, get_model_config

def main(config=None, writer=None, device=None):

    encoder_name, encoder_weights, in_channels, classes = get_model_config(config)

    # Create dataset and dataloader for evaluation
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

    # Initialize the model with the same parameters as in training
    model = get_unet_mobilenet_v3(
        in_channels=in_channels,
        classes=classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )
    model = model.to(device)

    # Construct the model path (saved under models/<encoder_name>/...)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "segmentation", encoder_name, f"{encoder_name}.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print(f"Please train the model with encoder '{encoder_name}' first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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

            # Log batch-level metrics
            writer.add_scalar("Eval/Batch_IoU", batch_iou, batch_idx)
            writer.add_scalar("Eval/Batch_DiceLoss", batch_dice, batch_idx)

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    print(f"Average IoU: {avg_iou:.4f}, Average Dice Loss: {avg_dice:.4f}")
    writer.add_scalar("Eval/Avg_IoU", avg_iou)
    writer.add_scalar("Eval/Avg_DiceLoss", avg_dice)

if __name__ == "__main__":
    # Load configuration using our config helper
    config = load_config(job="evaluantion", step="segmentation")

    # Setup TensorBoard logger for evaluation in a dedicated subfolder
    writer = init_logger(config=config)

    # Default is cuda if available, else cpu
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    main(config=config, writer=writer, device=device)

    writer.close()
