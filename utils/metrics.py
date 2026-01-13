"""
Evaluation metrics for medical AI models
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch


def calculate_metrics(y_true, y_pred, y_scores=None, average='macro'):
    """
    Calculate comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Predicted probabilities (for AUC)
        average: Averaging method for multi-class metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    # Calculate AUC if scores are provided
    if y_scores is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores, average=average, multi_class='ovr')
        except:
            metrics['auc'] = None

    # Calculate sensitivity and specificity for binary classification
    if len(np.unique(y_true)) == 2:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def print_classification_report(y_true, y_pred, class_names):
    """
    Print detailed classification report

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))


def calculate_class_wise_metrics(y_true, y_pred, class_names):
    """
    Calculate metrics for each class separately

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Dictionary with per-class metrics
    """
    n_classes = len(class_names)
    class_metrics = {}

    for i, class_name in enumerate(class_names):
        # Create binary labels for this class
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)

        class_metrics[class_name] = {
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }

    return class_metrics


def balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy (useful for imbalanced datasets)

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Balanced accuracy score
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)
    return np.mean(per_class)


def evaluate_model(model, data_loader, device='cuda', return_predictions=False):
    """
    Evaluate model on a dataset

    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        return_predictions: Whether to return predictions and labels

    Returns:
        Metrics dictionary and optionally predictions and labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    metrics = calculate_metrics(all_labels, all_preds, all_scores)

    if return_predictions:
        return metrics, all_preds, all_labels, all_scores
    else:
        return metrics
