"""
Visualization utilities for medical AI models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_true, y_scores, n_classes, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification

    Args:
        y_true: True labels (one-hot encoded)
        y_scores: Predicted probabilities
        n_classes: Number of classes
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_gradcam(model, image, target_layer, class_idx=None):
    """
    Visualize GradCAM for model interpretability

    Args:
        model: PyTorch model
        image: Input image tensor
        target_layer: Layer to compute gradients
        class_idx: Target class index (if None, use predicted class)

    Returns:
        GradCAM heatmap
    """
    # TODO: Implement GradCAM visualization
    pass


def plot_sample_predictions(images, true_labels, pred_labels, class_names, n_samples=9):
    """
    Plot sample predictions with true and predicted labels

    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to plot
    """
    n_samples = min(n_samples, len(images))
    rows = int(np.sqrt(n_samples))
    cols = int(np.ceil(n_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()

    for idx in range(n_samples):
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].axis('off')

        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]
        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'

        axes[idx].set_title(f'True: {true_class}\nPred: {pred_class}', color=color)

    plt.tight_layout()
    plt.show()
