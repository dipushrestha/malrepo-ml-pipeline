"""Data augmentation and transformation pipelines."""
from __future__ import annotations


def get_train_transforms(image_size: int = 224):
    """Get training transforms with augmentation."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except ImportError:
        return None


def get_val_transforms(image_size: int = 224):
    """Get validation/test transforms (no augmentation)."""
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    except ImportError:
        return None


def get_inference_transforms(image_size: int = 224):
    """Get inference transforms (same as validation)."""
    return get_val_transforms(image_size)
