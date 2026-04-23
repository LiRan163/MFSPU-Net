import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import os
from PIL import Image
import pandas as pd
from metric import calculate_iou, calculate_class_pa

VOC_COLORMAP = [
    (0, 0, 0),  # 0=background
    (128, 0, 0),  # 1=aeroplane
    (0, 128, 0),  # 2=bicycle
    (128, 128, 0),  # 3=bird
    (0, 0, 128),  # 4=boat
    (128, 0, 128),  # 5=bottle
    (0, 128, 128),  # 6=bus
    (128, 128, 128),  # 7=car
    (64, 0, 0),  # 8=cat
    (192, 0, 0),  # 9=chair
    (64, 128, 0),  # 10=cow
    (192, 128, 0),  # 11=diningtable
    (64, 0, 128),  # 12=dog
    (192, 0, 128),  # 13=horse
    (64, 128, 128),  # 14=motorbike
    (192, 128, 128),  # 15=person
    (0, 64, 0),  # 16=potted plant
    (128, 64, 0),  # 17=sheep
    (0, 192, 0),  # 18=sofa
    (128, 192, 0),  # 19=train
    (0, 64, 128)  # 20=tv/monitor
]

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "potted plant", "sheep", "sofa", "train",
    "tv/monitor"
]


def color2label(pred_img):
    """Convert color prediction map to grayscale class map"""
    pred_np = np.array(pred_img)
    h, w, _ = pred_np.shape
    label = np.zeros((h, w), dtype=np.int64)

    colormap2label = {color: idx for idx, color in enumerate(VOC_COLORMAP)}
    for color, idx in colormap2label.items():
        mask = np.all(pred_np == np.array(color), axis=-1)
        label[mask] = idx
    return label


def evaluate_folder(pred_dir, label_dir, num_classes=21):
    """
    Traverse the prediction result folder, calculate mean mIoU and Pixel Accuracy,
    display IoU for each class, and save results to CSV (horizontal output)
    """
    ious, accs = [], []
    class_ious = [[] for _ in range(num_classes)]

    for filename in os.listdir(pred_dir):
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            continue

        pred_img = Image.open(os.path.join(pred_dir, filename)).convert("RGB")
        pred_label = color2label(pred_img)

        gt_path = os.path.join(label_dir, filename.replace(".jpg", ".png"))
        if not os.path.exists(gt_path):
            print(f"⚠️ Label missing: {gt_path}")
            continue
        gt_img = Image.open(gt_path)
        gt_label = np.array(gt_img, dtype=np.int64)

        pred_tensor = torch.from_numpy(pred_label).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_label)

        pred_onehot = torch.nn.functional.one_hot(
            pred_tensor, num_classes=num_classes
        ).permute(0, 3, 1, 2).float()

        iou = calculate_iou(pred_onehot, gt_tensor, num_classes)
        acc = calculate_class_pa(pred_onehot, gt_tensor, num_classes)

        preds_flat = torch.argmax(pred_onehot, dim=1).flatten()
        labels_flat = gt_tensor.flatten()
        cm = confusion_matrix(
            labels_flat.cpu().numpy(),
            preds_flat.cpu().numpy(),
            labels=np.arange(num_classes)
        )
        for cls in range(num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            denom = tp + fp + fn
            if denom > 0:
                class_ious[cls].append(tp / denom)

        ious.append(iou)
        accs.append(acc)

    mean_iou = np.mean(ious)
    mean_acc = np.mean(accs)

    print(f"\n===== Final Results =====")
    print(f"Mean mIoU: {mean_iou * 100:.2f}")
    print(f"Mean Pixel Acc: {mean_acc * 100:.2f}\n")

    print("IoU per class (%):")
    class_iou_means = []
    for cls, values in enumerate(class_ious):
        if len(values) > 0:
            iou_val = np.mean(values) * 100
            class_iou_means.append(iou_val)
            print(f"{VOC_CLASSES[cls]:<12}: {iou_val:.2f}")
        else:
            class_iou_means.append(np.nan)
            print(f"{VOC_CLASSES[cls]:<12}: No valid pixels")

    results = {VOC_CLASSES[i]: [round(class_iou_means[i], 2)] for i in range(num_classes)}
    results["Mean IoU"] = [round(mean_iou * 100, 2)]
    results["Pixel Acc"] = [round(mean_acc * 100, 2)]

    df = pd.DataFrame(results)

    csv_path = os.path.join(os.path.dirname(pred_dir), "metrics_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Results saved to: {csv_path}")

    return mean_iou, mean_acc


if __name__ == "__main__":
    pred_dir = r"C:\Users\China\Desktop\contrastive model\ours\results"
    label_dir = r"E:\datasets\PASCAL_VOC_2012\VOC2012_train_val\VOC2012_train_val\test\label"

    evaluate_folder(pred_dir, label_dir, num_classes=21)
