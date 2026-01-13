"""
Data utility functions for loading and preprocessing medical data
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images"""

    def __init__(self, data_dir, transform=None, label_file=None):
        """
        Args:
            data_dir: Directory with all the images
            transform: Optional transform to be applied on images
            label_file: CSV file with image labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        if label_file:
            self.labels_df = pd.read_csv(label_file)
            self.image_files = self.labels_df['filename'].tolist()
            self.labels = self.labels_df['label'].tolist()
        else:
            self.image_files = list(self.data_dir.glob('*.jpg')) + \
                              list(self.data_dir.glob('*.png'))
            self.labels = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.data_dir / self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels:
            label = self.labels[idx]
            return image, label
        else:
            return image


def get_transforms(image_size=224, augment=True):
    """
    Get image transforms for training and validation

    Args:
        image_size: Size to resize images to
        augment: Whether to apply data augmentation

    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return {'train': train_transform, 'val': val_transform}


def handle_class_imbalance(dataset, method='weighted_loss'):
    """
    Handle class imbalance in medical datasets

    Args:
        dataset: PyTorch dataset
        method: Method to handle imbalance ('weighted_loss', 'oversample', 'undersample')

    Returns:
        Class weights or sampler depending on method
    """
    if not hasattr(dataset, 'labels'):
        raise ValueError("Dataset must have labels attribute")

    labels = np.array(dataset.labels)
    class_counts = np.bincount(labels)

    if method == 'weighted_loss':
        # Calculate class weights for weighted loss
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)

    elif method == 'oversample':
        # TODO: Implement oversampling
        pass

    elif method == 'undersample':
        # TODO: Implement undersampling
        pass

    return None
