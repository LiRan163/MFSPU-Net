import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_iou(preds, labels, num_classes, ignore_label=255):
    """
    Calculate IoU for each class and global mIoU, complying with evaluation rules
    Args:
        preds: [B, C, H, W] - Model output logits (without softmax)
        labels: [B, H, W] - Ground truth labels (containing ignore_label)
        num_classes: Total number of classes
        ignore_label: Value of the label to ignore (default 255)
    Returns:
        miou: Global mean Intersection over Union
        class_ious: List of IoU for each class
    """
    preds = torch.argmax(preds, dim=1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    valid_mask = (labels_flat != ignore_label)
    preds_valid = preds_flat[valid_mask].cpu().numpy()
    labels_valid = labels_flat[valid_mask].cpu().numpy()

    cm = confusion_matrix(
        labels_valid,
        preds_valid,
        labels=np.arange(num_classes)
    )

    class_ious = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        denom = tp + fp + fn

        if denom == 0:
            class_ious.append(1.0)
        else:
            class_ious.append(tp / denom)

    miou = np.mean(class_ious)
    return miou, class_ious


def calculate_class_pa(preds, labels, num_classes, ignore_label=255):
    """
    Calculate PA for each class and global mPA, complying with evaluation rules (replaces the original pixel_accuracy function)
    Args:
        preds: [B, C, H, W] - Model output logits (without softmax)
        labels: [B, H, W] - Ground truth labels (containing ignore_label)
        num_classes: Total number of classes
        ignore_label: Value of the label to ignore (default 255)
    Returns:
        mpa: Global mean Pixel Accuracy
        class_pas: List of PA for each class
    """
    preds = torch.argmax(preds, dim=1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    valid_mask = (labels_flat != ignore_label)
    preds_valid = preds_flat[valid_mask].cpu().numpy()
    labels_valid = labels_flat[valid_mask].cpu().numpy()

    cm = confusion_matrix(
        labels_valid,
        preds_valid,
        labels=np.arange(num_classes)
    )

    class_pas = []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        total = cm[cls, :].sum()

        if total == 0:
            class_pas.append(1.0)
        else:
            class_pas.append(tp / total)

    mpa = np.mean(class_pas)
    return mpa, class_pas

# def main():
#     preds = torch.randn(2, 19, 512, 1024)  # Model logits output
#     # labels = torch.randint(0, 19, (2, 512, 1024))  # Ground truth labels (when no ignore label)
#     labels = torch.argmax(preds, dim=1)
#     print(preds.shape, labels.shape)
#
#     miou, class_ious = calculate_iou(preds, labels, num_classes=19, ignore_label=255)
#     mpa, class_pas = calculate_class_pa(preds, labels, num_classes=19, ignore_label=255)
#
#     print(f"mIoU: {miou:.4f}")
#     print(f"mPA: {mpa:.4f}")
#     print("IoU for each class:", [round(iou, 4) for iou in class_ious])
#     print("PA for each class:", [round(pa, 4) for pa in class_pas])
#
#
# if __name__ == "__main__":
#     main()
