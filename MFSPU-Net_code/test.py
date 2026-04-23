import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix
from dataloader import create_segmentation_dataloader
# from dataloader_noresize import create_segmentation_dataloader
from unet import UNet
from metric import calculate_iou, calculate_class_pa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SegmentationTester:
    """Tester class responsible for model testing and result visualization, ensuring all operations run on GPU"""

    def __init__(self, model, test_loader, num_classes, model_path='model_sgd/best_unet_model_miou.pth', device=device):

        # if device != 'cuda' or not torch.cuda.is_available():
        #     raise ValueError("Current configuration requires GPU support, but no available GPU device detected")

        self.device = device
        self.model = model.to(self.device)
        self.test_loader = test_loader
        self.num_classes = num_classes

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.color_map = self._create_color_map()

    def _create_color_map(self):
        """Create class to color mapping"""
        np.random.seed(42)
        color_map = np.random.randint(0, 256, size=(self.num_classes, 3), dtype=np.uint8)
        color_map[0] = [0, 0, 0]  # Set background to black
        return color_map

    def visualize_results(self, num_samples=5):
        """Visualize test results"""
        samples_visualized = 0
        plt.figure(figsize=(15, 5 * num_samples))

        with torch.no_grad():
            for images, labels in self.test_loader:
                if samples_visualized >= num_samples:
                    break

                images = images.to(self.device, non_blocking=True)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                image_np = images[0].cpu().numpy().transpose(1, 2, 0)
                label_np = labels[0].cpu().numpy()
                pred_np = preds[0].cpu().numpy()

                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1)

                label_color = self.color_map[label_np]
                pred_color = self.color_map[pred_np]

                samples_visualized += 1
                plt.subplot(num_samples, 3, (samples_visualized - 1) * 3 + 1)
                plt.imshow(image_np)
                plt.title('input image')
                plt.axis('off')

                plt.subplot(num_samples, 3, (samples_visualized - 1) * 3 + 2)
                plt.imshow(label_color)
                plt.title('true label')
                plt.axis('off')

                plt.subplot(num_samples, 3, (samples_visualized - 1) * 3 + 3)
                plt.imshow(pred_color)
                plt.title('predicted result')
                plt.axis('off')

        plt.tight_layout()
        plt.savefig('test_image/segmentation_results.png')
        # plt.show()

        torch.cuda.empty_cache()

    def evaluate(self):
        """Evaluate model performance on test set"""
        print(f'Start evaluation, using device: {self.device} ({torch.cuda.get_device_name(0)})')
        all_miou = 0.0
        all_pixel_acc = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
            for batch_idx, (images, labels) in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)

                # print(outputs.shape, labels.shape)

                miou = calculate_iou(outputs, labels, self.num_classes)
                all_miou += miou

                pixel_acc = calculate_class_pa(outputs, labels, self.num_classes)
                all_pixel_acc += pixel_acc

                pbar.set_description(f'Evaluation, Batch {batch_idx + 1}/{len(self.test_loader)}')
                pbar.set_postfix(miou=miou, pixel_acc=pixel_acc)

                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()

        avg_miou = all_miou / len(self.test_loader)
        avg_pixel_acc = all_pixel_acc / len(self.test_loader)

        print(f'\nEvaluation results:')
        print(f'Average mIoU: {avg_miou:.4f}')
        print(f'Average pixel accuracy: {avg_pixel_acc:.4f}')

        torch.cuda.empty_cache()

        return avg_miou, avg_pixel_acc


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This code must run in an environment with GPU support, but no available GPU device detected")

    config = {
        "image_dir": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/JPEGImages",
        "label_dir": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/SegmentationClass_Gray",
        "train_split": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt",
        "val_split": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/test.txt",
        "batch_size": 1,
        "num_workers": 4,
        "pin_memory": True,  # Enable memory pinning to accelerate data transfer to GPU
        "img_size": (512, 512),
        "num_classes": 21,
        "num_epochs": 300
    }

    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")

    print("Creating validation dataloader...")
    val_loader = create_segmentation_dataloader(
        image_dir=config["image_dir"],
        label_dir=config["label_dir"],
        split_file=config["val_split"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        is_train=False
    )

    model = UNet(n_channels=3, n_classes=config["num_classes"], bilinear=True)
    print('The total number of parameters', count_parameters(model))

    # Test model
    print("Start testing model...")
    tester = SegmentationTester(
        model=model,
        test_loader=val_loader,
        # Use validation set as test set here, independent test set should be used in practical applications
        num_classes=config["num_classes"],
        device=device
    )
    tester.evaluate()
    tester.visualize_results(num_samples=5)
