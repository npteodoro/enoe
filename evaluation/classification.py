import torch
import torchvision.transforms as transforms

from data.loaders.classification import ClassificationDataset
from jobs.evaluation import EvaluationStep

class EvaluationClassification(EvaluationStep):

    def __init__(self, config, logger):
        """
        Initialize step with configuration and logger.
        """
        super().__init__(config, logger)
        self.config = config
        self.logger = logger

        self.use_mask = self.config_model.get("use_mask", False)  # Add this line
        self.encoder_name = self.config_model.get("encoder_name", "mobilenetv3_small_classifier")

    def define_transform(self):
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
            mask_folder=self.config_dataset.get("mask_folder", "mask") if self.use_mask else None
        )

    def evaluate(self):

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.dataloader:
                if self.use_mask:
                    images, masks, labels = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images, masks)
                else:
                    images, _, labels = batch  # Ignore mask
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)

                _, preds = torch.max(outputs, 1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Classification Accuracy: {accuracy:.4f}")
        self.logger.add_scalar("Eval/Accuracy", accuracy)
