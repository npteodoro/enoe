# data/augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_augmentations(image_size=(256,256)):
    return A.Compose([
        A.Resize(*image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_augmentations(image_size=(256,256)):
    return A.Compose([
        A.Resize(*image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

