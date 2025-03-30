import torch
import torchvision.transforms as transforms

from architectures.segmentation.segmentation_unet import get_unet_mobilenet_v3
from data.loaders.segmentation import RiverSegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss

from jobs.evaluation import EvaluationStep

class EvaluationSegmentation(EvaluationStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

    def define_transform(self):
        """
        Define the transformations for the dataset.
        """
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.config_dataset.get("image_size"))),
            transforms.ToTensor(),
        ])

    def define_dataset(self):
        """
        Define the dataset configuration.
        """
        self.dataset = RiverSegmentationDataset(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            rgb_folder=self.config_dataset.get("rgb_folder", "rgb"),
            mask_folder=self.config_dataset.get("mask_folder", "mask"),
            image_size=tuple(self.config_dataset.get("image_size")),
            transform=self.transform
        )

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        self.model = get_unet_mobilenet_v3(
            in_channels=self.config_model.get("in_channels", 3),
            classes=self.config_model.get("classes", 1),
            encoder_name=self.config_model.get("encoder_name", "timm-mobilenetv3_small_100"),
            encoder_weights=self.config_model.get("encoder_weights", "imagenet")
        ).to(self.device)

    def run_model(self):
        """
        Run the model on a sample batch.
        """

        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = torch.sigmoid(self.model(images))
                batch_iou = iou_score(outputs, masks)
                batch_dice = dice_loss(outputs, masks)

                total_iou += batch_iou
                total_dice += batch_dice
                num_batches += 1

                # Log batch-level metrics
                self.logger.add_scalar("Eval/Batch_IoU", batch_iou, batch_idx)
                self.logger.add_scalar("Eval/Batch_DiceLoss", batch_dice, batch_idx)

        avg_iou = total_iou / num_batches
        avg_dice = total_dice / num_batches
        print(f"Average IoU: {avg_iou:.4f}, Average Dice Loss: {avg_dice:.4f}")
        self.logger.add_scalar("Eval/Avg_IoU", avg_iou)
        self.logger.add_scalar("Eval/Avg_DiceLoss", avg_dice)
