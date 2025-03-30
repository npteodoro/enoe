# training/train_segmentation.py
import os
import torch
from torch.utils.data import DataLoader
from architectures.segmentation.segmentation_unet import get_unet_mobilenet_v3
from data.loaders.segmentation import RiverSegmentationDataset
from utils.evaluation_metrics import iou_score, dice_loss
import torchvision.transforms as transforms

from jobs.base import Step

class TrainingSegmentation(Step):

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
            transforms.Resize(tuple(self.config_dataset["image_size"])),
            transforms.ToTensor(),
        ])

    def define_dataset(self):
        """
        Define the dataset configuration.
        """
        self.dataset = RiverSegmentationDataset(
            csv_file=self.config_dataset["csv_file"],
            root_dir=self.config_dataset["root_dir"],
            rgb_folder=self.config_dataset.get("rgb_folder", "rgb"),
            mask_folder=self.config_dataset.get("mask_folder", "mask"),
            image_size=tuple(self.config_dataset["image_size"]),
            transform=self.transform
        )

    def define_dataloader(self):
        """
        Define the dataloader configuration.
        """
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config_training["batch_size"],
            shuffle=True,
            num_workers=self.config_training["num_workers"]
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
        )
        self.model = self.model.to(self.device)

    def save_model(self):
        """
        Save model checkpoint in a subfolder for the current encoder
        """
        model_dir = os.path.join(self.config.get_root_dir(), "models", "segmentation", self.encoder_name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, f"{self.encoder_name}.pth"))
        print("Training complete and model saved.")

    def run_model(self):
        """
        Run the model on a sample batch.
        """
        # Loss and optimizer
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_training["learning_rate"])

        num_epochs = self.config_training["num_epochs"]
        total_samples = len(self.dataset)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, masks in self.dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)  # [B, 1, H, W]
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, IoU: {iou:.4f}, Dice Loss: {dice:.4f}")

            self.save_model()

    def process(self):

        # Create dataset and dataloader
        self.define_transform()

        self.define_dataset()

        self.define_dataloader()

        self.initialize_model()

        self.run_model()
