import torch
import torch.nn as nn
import torchvision.transforms as transforms

from architectures.classification.classifier_dual_input import get_dual_input_model
from data.loaders.classification import ClassificationDataset
from jobs.training import TrainingStep

class TrainingClassification(TrainingStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

        self.use_mask = self.config_model.get("use_mask", False)  # Add a config option for this

        self.encoder_name = self.config_model.get("encoder_name", "mobilenetv3_small_classifier")

    def define_transform(self):
        """
        Define the transformations for the dataset.
        """
        self.transform = transforms.Compose([
            transforms.Resize(tuple(self.config_dataset.get("image_size"))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def define_dataset(self):
        """
        Define the dataset configuration.
        """
        self.dataset = ClassificationDataset(
            csv_file=self.config_dataset.get("csv_file"),
            root_dir=self.config_dataset.get("root_dir"),
            rgb_folder=self.config_dataset.get("rgb_folder", "rgb"),
            mask_folder=self.config_dataset.get("mask_folder", "mask") if self.use_mask else None,
            transform=self.transform
        )

    def initialize_model(self):
        """
        Initialize the model configuration.
        """
        self.model = get_dual_input_model(
            backbone_name=self.config_model.get("backbone_name", "shufflenet"),
            num_classes=self.config_model.get("num_classes", 4),  # e.g., 4 classes: low, medium, high, floo,
            pretrained=True
        ).to(self.device)

    def run_model(self):
        """
        Run the model training.
        """
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config_training.get("learning_rate"))

        num_epochs = self.config_training.get("num_epochs")
        total_samples = len(self.dataloader.dataset)
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0

            # Current code has incorrect unpacking - should handle masks too
            if self.use_mask:
                for (images, masks, labels) in self.dataloader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images, masks)  # Pass both images and masks
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels).item()

                    self.logger.add_scalar("Train/BatchLoss", loss.item(), global_step)
                    global_step += 1
            else:
                for (images, _, labels) in self.dataloader:  # Ignore masks
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += torch.sum(preds == labels).item()

                    self.logger.add_scalar("Train/BatchLoss", loss.item(), global_step)
                    global_step += 1

            epoch_loss = running_loss / total_samples
            epoch_acc = correct / total_samples
            self.logger.add_scalar("Train/EpochLoss", epoch_loss, epoch)
            self.logger.add_scalar("Train/EpochAccuracy", epoch_acc, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
