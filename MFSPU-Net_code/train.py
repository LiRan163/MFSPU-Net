import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.optim.lr_scheduler import _LRScheduler, StepLR

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model(model, init_type='nor'):
    """
    Uniformly initialize all learnable parameters in the model
    """
    for name, param in model.named_parameters():
        if init_type == 'zero':
            nn.init.zeros_(param.data)
        elif init_type == 'nor':
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0.0, std=0.01)
            elif 'bias' in name:
                torch.fill_(param.data, value=0.0)
        else:
            raise ValueError(f"Unsupported initialization type: {init_type}, options are 'zero' or 'nor'")


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


class SegmentationTrainer:
    """Trainer class responsible for model training and validation, ensuring all operations run on GPU"""

    def __init__(self, model, train_loader, val_loader, num_classes, lr, weight_decay, patience, optimizer_type,
                 loss_type, if_class, device):
        self.patience = patience
        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.if_class = if_class
        self.device = device
        self.count_patience = 0
        self.alpha = 0.1
        print(self.device, torch.cuda.is_available())
        # if self.device != 'cuda' or not torch.cuda.is_available():
        #     raise ValueError("Current configuration requires GPU support, but no available GPU device detected")

        self.model = model.to(self.device)
        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}, Parameter shape: {param.shape}")
        assert next(self.model.parameters()).is_cuda, "Model not successfully loaded to GPU"
        print("Model successfully loaded to GPU")
        # self.model = self.model.cuda()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes

        # weights = torch.ones(self.num_classes)
        # weights[0] = 0.1
        #
        # self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device)).to(self.device)

        if self.loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(self.device)
        else:
            self.criterion = nn.NLLLoss2d().to(self.device)
        if self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
        else:
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        self.cls_criterion = nn.BCEWithLogitsLoss().to(self.device)

        self.scheduler = PolyLR(self.optimizer, max_iters=30000, power=0.9)

        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=20, factor=0.5
        # )

        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.val_accuracy = []
        self.best_miou = 0.0
        self.best_accuracy = 0.0
        self.best_val_loss = float('inf')
        self.best_model_path_acc = 'model/best_unet_model_acc.pth'
        self.best_model_path_loss = 'model/best_unet_model_loss.pth'
        self.best_model_path_miou = 'model/best_unet_model_miou.pth'

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch_idx, (images, labels, y_cls, phase_image, gradient_image, hsv_image) in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            phase_image = phase_image.to(self.device)
            gradient_image = gradient_image.to(self.device)
            hsv_image = hsv_image.to(self.device)

            self.optimizer.zero_grad()

            if self.if_class:
                y_cls = y_cls.to(self.device)
                outputs, out_class = self.model(images, phase_image, gradient_image, hsv_image)
                if self.loss_type == 'CE':
                    loss = self.criterion(outputs, labels) + self.alpha * self.cls_criterion(out_class, y_cls)
                else:
                    log_outputs = F.log_softmax(outputs, dim=1)
                    loss = self.criterion(log_outputs, labels) + self.alpha * self.cls_criterion(out_class, y_cls)
            else:
                outputs = self.model(images, phase_image, gradient_image, hsv_image)
                if self.loss_type == 'CE':
                    loss = self.criterion(outputs, labels)
                else:
                    log_outputs = F.log_softmax(outputs, dim=1)
                    loss = self.criterion(log_outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            pbar.set_description(f'Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(self.train_loader)}')
            pbar.set_postfix(loss=loss.item(), gpu=torch.cuda.get_device_name(0))

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        """Evaluate model on validation set"""
        self.model.eval()
        running_loss = 0.0
        all_miou = 0.0
        all_accuracy = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for batch_idx, (images, labels, _, phase_image, gradient_image, hsv_image) in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                phase_image = phase_image.to(self.device)
                gradient_image = gradient_image.to(self.device)
                hsv_image = hsv_image.to(self.device)

                if self.if_class:
                    outputs, _ = self.model(images, phase_image, gradient_image, hsv_image)
                else:
                    outputs = self.model(images, phase_image, gradient_image, hsv_image)

                if self.loss_type == 'CE':
                    loss = self.criterion(outputs, labels)
                else:
                    log_outputs = F.log_softmax(outputs, dim=1)
                    loss = self.criterion(log_outputs, labels)

                running_loss += loss.item()

                miou = calculate_iou(outputs, labels, self.num_classes)
                accuracy = calculate_class_pa(outputs, labels, self.num_classes)
                all_miou += miou
                all_accuracy += accuracy

                pbar.set_description(f'Validation, Batch {batch_idx + 1}/{len(self.val_loader)}')
                pbar.set_postfix(loss=loss.item(), miou=miou)

        avg_loss = running_loss / len(self.val_loader)
        avg_miou = all_miou / len(self.val_loader)
        avg_accuracy = all_accuracy / len(self.val_loader)

        self.val_losses.append(avg_loss)
        self.val_mious.append(avg_miou)
        self.val_accuracy.append(avg_accuracy)

        if avg_loss <= self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(), self.best_model_path_loss)
            print(f'Saved best model, Val Loss: {avg_loss:.4f}')
            self.count_patience = 0
        else:
            self.count_patience += 1

        if avg_miou > self.best_miou:
            self.best_miou = avg_miou
            torch.save(self.model.state_dict(), self.best_model_path_miou)
            print(f'Saved best model, MIoU: {avg_miou:.4f}')

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            torch.save(self.model.state_dict(), self.best_model_path_acc)
            print(f'Saved best model, Pixel Accuracy: {avg_accuracy:.4f}')

        return avg_loss, avg_miou, avg_accuracy, self.count_patience

    def train(self, num_epochs):
        """Main function for model training"""
        print(f'Start training, using device: {self.device} ({torch.cuda.get_device_name(0)})')
        print(f'GPU Memory: Total {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB')
        start_time = time.time()

        # Enable cuDNN benchmark for acceleration
        torch.backends.cudnn.benchmark = True

        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            print('-' * 50)

            train_loss = self.train_epoch(epoch)
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Current GPU Memory Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB')

            val_loss, val_miou, val_accuracy, count_pat = self.validate()
            print(
                f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, Val Pixel Accuracy: {val_accuracy:.4f}, Count Patience: {count_pat}')

            self.scheduler.step()
            # self.scheduler.step(val_loss)

            if count_pat > self.patience:
                print("Training stopped early!")
                break

            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f'\nTraining completed! Total time: {total_time:.2f} seconds')
        print(f'Best validation mIoU: {self.best_miou:.4f}')

        self.plot_training_curves()

        # Release GPU memory
        torch.cuda.empty_cache()

    def plot_training_curves(self):
        """Plot loss and mIoU curves during training and save to image folder"""
        os.makedirs("image", exist_ok=True)

        plt.figure(figsize=(15, 8))

        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='train_loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(range(1, len(self.val_mious) + 1), self.val_mious, label='val_mIoU')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Validation Set mIoU Curve')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(range(1, len(self.val_accuracy) + 1), self.val_accuracy, label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('pixel accuracy')
        plt.title('Validation Set Pixel Accuracy Curve')
        plt.legend()

        plt.tight_layout()
        plt.savefig('image/training_curves.png')
        # plt.show()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This code must run in an environment with GPU support, but no available GPU device detected")

    config = {
        "image_dir": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/JPEGImages",
        "label_dir": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/SegmentationClass_Gray",
        "train_split": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt",
        "val_split": "/mnt/ssd4/hecongbing/datasets/PASCAL_VOC_2012/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/val.txt",
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,  # Enable memory pinning to accelerate data transfer to GPU
        "img_size": (512, 512),
        "num_classes": 21,
        "num_epochs": 30000,
        "bilinear": True,
        "init_type": "nor",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "patience": 30000,
        "loss_type": "CE",
        "optimizer_type": "SGD",
        "if_pre": False,
        "if_class": False
    }

    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")

    print("Creating training dataloader...")
    train_loader = create_segmentation_dataloader(
        image_dir=config["image_dir"],
        label_dir=config["label_dir"],
        split_file=config["train_split"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        is_train=True
    )

    print("Creating validation dataloader...")
    val_loader = create_segmentation_dataloader(
        image_dir=config["image_dir"],
        label_dir=config["label_dir"],
        split_file=config["val_split"],
        img_size=config["img_size"],
        batch_size=1,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        is_train=False
    )

    model = UNet(n_channels=3, n_classes=config["num_classes"], bilinear=config["bilinear"], if_pre=config["if_pre"],
                 if_class=config["if_class"])
    initialize_model(model=model, init_type=config["init_type"])
    # print("Model parameters:")
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         print(f"Parameter name: {name}, Weight shape: {param.shape}, First 10 values: {param.data.flatten()[:50]}")
    #     elif 'bias' in name:
    #         print(f"Parameter name: {name}, Bias shape: {param.shape}, First 10 values: {param.data.flatten()[:10]}")

    print("Starting model training...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=config["num_classes"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
        optimizer_type=config["optimizer_type"],
        loss_type=config["loss_type"],
        if_class=config["if_class"],
        device=device
    )
    trainer.train(num_epochs=config["num_epochs"])
