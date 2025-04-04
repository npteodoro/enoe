import torch
import torchvision.transforms as transforms

# from architectures.segmentation.segmentation_unet import get_unet_mobilenet_v3
from data.loaders.segmentation import SegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss
from jobs.training import TrainingStep

class TrainingSegmentation(TrainingStep):

    def __init__(self, config, logger):
        """
        Initialize the TrainingSegmentation step with configuration and logger.
        """
        super().__init__(config, logger)

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
        self.dataset = SegmentationDataset(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            rgb_folder=self.config_dataset.get("rgb_folder", "rgb"),
            mask_folder=self.config_dataset.get("mask_folder", "mask"),
            image_size=tuple(self.config_dataset.get("image_size")),
            transform=self.transform
        )

    def define_dataloader(self):
        super().define_dataloader(shuffle=True)

    # def initialize_model(self):
    #     """
    #     Initialize the model configuration.
    #     """
    #     self.model = get_unet_mobilenet_v3(
    #         in_channels=self.config_model.get("in_channels", 3),
    #         classes=self.config_model.get("classes", 1),
    #         encoder_name=self.config_model.get("encoder_name", "timm-mobilenetv3_small_100"),
    #         encoder_weights=self.config_model.get("encoder_weights", "imagenet")
    #     )
    #     self.model = self.model.to(self.device)

    def run_model(self):
        """
        Run the model on a sample batch.
        """
        num_epochs = self.config_training.get("num_epochs")
        total_samples = len(self.dataset)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, masks in self.dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)  # [B, 1, H, W]
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)
                self.logger.add_scalar("Train/BatchLoss", loss.item(), global_step)
                global_step += 1

            epoch_loss = running_loss / total_samples

            # Evaluate on a small validation batch (using training data here for simplicity)
            self.model.eval()
            with torch.no_grad():
                sample_images, sample_masks = next(iter(self.dataloader))
                sample_images = sample_images.to(self.device)
                sample_masks = sample_masks.to(self.device)
                sample_outputs = torch.sigmoid(self.model(sample_images))
                iou = iou_score(sample_outputs, sample_masks)
                dice = dice_loss(sample_outputs, sample_masks)

            self.logger.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            self.logger.add_scalar("Train/IoU", iou, epoch)
            self.logger.add_scalar("Train/DiceLoss", dice, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, "
                  f"IoU: {iou:.4f}, Dice Loss: {dice:.4f}")
