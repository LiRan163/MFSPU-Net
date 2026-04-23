import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = 21


def compute_phase_image(pil_image):
    """
    Input:
        pil_image: PIL.Image RGB image
    Output:
        phase_tensor: torch.Tensor, shape=(3,H,W), phase spectrum normalized to [0,1]
    """
    image_np = np.array(pil_image).astype(np.float32)

    phase_np = np.zeros_like(image_np, dtype=np.float32)

    for c in range(3):
        f = np.fft.fft2(image_np[:, :, c])
        fshift = np.fft.fftshift(f)
        phase_np[:, :, c] = np.angle(fshift)

    phase_np = (phase_np + np.pi) / (2 * np.pi)

    phase_image = Image.fromarray((phase_np * 255).astype(np.uint8))
    phase_tensor = transforms.ToTensor()(phase_image)

    return phase_tensor


def compute_gradient_image(pil_image):
    """
    Input:
        pil_image: PIL.Image RGB image
    Output:
        gradient_tensor: torch.Tensor, shape=(3,H,W), gradient magnitude for each channel normalized to [0,1]
    """
    image_np = np.array(pil_image).astype(np.float32)

    gradient_np = np.zeros_like(image_np, dtype=np.float32)

    for c in range(3):
        channel = image_np[:, :, c]
        grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        gradient_np[:, :, c] = grad_mag

    gradient_np = (gradient_np - gradient_np.min()) / (gradient_np.max() - gradient_np.min() + 1e-8)

    gradient_image = Image.fromarray((gradient_np * 255).astype(np.uint8))
    gradient_tensor = transforms.ToTensor()(gradient_image)

    return gradient_tensor


def get_hsv_image(pil_image):
    """
    Input:
        pil_image: PIL.Image RGB image
    Output:
        hsv_tensor: torch.Tensor, shape=(3,H,W), values normalized to [0,1]
    """
    img_np = np.array(pil_image)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)

    hsv[:, :, 0] /= 179.0
    hsv[:, :, 1] /= 255.0
    hsv[:, :, 2] /= 255.0

    hsv_tensor = torch.from_numpy(hsv.transpose(2, 0, 1))

    return hsv_tensor


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, split_file, pt_dir,
                 img_size=(512, 512), transform_img=None, transform_label=None, transform_aux=None):

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size

        with open(split_file, 'r') as f:
            self.sample_names = [line.strip() for line in f if line.strip()]

        self.phase_data = torch.load(os.path.join(pt_dir, "phase.pt"))
        self.gradient_data = torch.load(os.path.join(pt_dir, "gradient.pt"))
        self.hsv_data = torch.load(os.path.join(pt_dir, "hsv.pt"))

        self.transform_img = transform_img if transform_img is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_label = transform_label if transform_label is not None else transforms.Compose([
            transforms.ToTensor()
        ])

        self.transform_aux = transform_aux if transform_aux is not None else transforms.Compose([])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        img_path = os.path.join(self.image_dir, f"{sample_name}.jpg")
        label_path = os.path.join(self.label_dir, f"{sample_name}.png")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"RGB image file does not exist: {img_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file does not exist: {label_path}")

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        phase_image = self.phase_data[sample_name]
        gradient_image = self.gradient_data[sample_name]
        hsv_image = self.hsv_data[sample_name]

        phase_image = self.transform_aux(phase_image)
        gradient_image = self.transform_aux(gradient_image)
        hsv_image = self.transform_aux(hsv_image)

        image = self.transform_img(image)
        label = self.transform_label(label)
        label = (label * 255).long().squeeze()

        assert image.shape[1:] == label.shape, \
            f"Image size {image.shape[1:]} does not match label size {label.shape}: {sample_name}"

        y_cls = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        unique_labels = torch.unique(label)
        for cls_id in unique_labels:
            if 0 <= cls_id < NUM_CLASSES:
                y_cls[cls_id] = 1

        return image, label, y_cls, phase_image, gradient_image, hsv_image


def create_segmentation_dataloader(
        image_dir,
        label_dir,
        split_file,
        pt_dir,
        img_size=(512, 512),
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        is_train=True
):
    if is_train:
        transform_img = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transform_label = transforms.Compose([
            transforms.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

        transform_aux = transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR)

    else:
        transform_img = None
        transform_label = None
        transform_aux = None

    dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        split_file=split_file,
        pt_dir=pt_dir,
        img_size=img_size,
        transform_img=transform_img,
        transform_label=transform_label,
        transform_aux=transform_aux
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return dataloader

# if __name__ == "__main__":
#     config = {
#         "image_dir": "E:\\datasets\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages",
#         "label_dir": "E:\\datasets\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\SegmentationClass_Gray",
#         "train_split": "E:\\datasets\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\ImageSets\\Segmentation\\train.txt",
#         "val_split": "E:\\datasets\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\ImageSets\\Segmentation\\val.txt",
#         "batch_size": 2,
#         "num_workers": 4,
#         "img_size": (512, 512)
#     }
#
#     print("Creating training dataloader...")
#     train_loader = create_segmentation_dataloader(
#         image_dir=config["image_dir"],
#         label_dir=config["label_dir"],
#         split_file=config["train_split"],
#         img_size=config["img_size"],
#         batch_size=config["batch_size"],
#         shuffle=True,
#         num_workers=config["num_workers"],
#         is_train=True
#     )
#
#     print("Creating validation dataloader...")
#     val_loader = create_segmentation_dataloader(
#         image_dir=config["image_dir"],
#         label_dir=config["label_dir"],
#         split_file=config["val_split"],
#         img_size=config["img_size"],
#         batch_size=1,
#         shuffle=False,
#         num_workers=config["num_workers"],
#         is_train=False
#     )
#
#     print(f"\nTraining set samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
#     print(f"Validation set samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")
#
#     try:
#         print("\nTesting first batch of training set...")
#         for batch_idx, (images, labels, y_cls, phase_image, gradient_image, hsv_image) in enumerate(train_loader):
#             print(
#                 f"Training batch {batch_idx + 1} - Image shape: {images.shape}, Label shape: {labels.shape}, Class label shape: {y_cls.shape}, "
#                 f"Phase spectrum shape: {phase_image.shape}, Gradient spectrum shape: {gradient_image.shape}, HSV image shape: {hsv_image.shape}")
#             print(f"Label class range: {labels.min().item()} ~ {labels.max().item()}")
#             print(f"Label classes: {torch.unique(labels)}")
#             print(f"Classes: {y_cls}")
#             break
#
#         print("\nTesting first batch of validation set...")
#         for batch_idx, (images, labels, y_cls, phase_image, gradient_image, hsv_image) in enumerate(val_loader):
#             print(
#                 f"Validation batch {batch_idx + 1} - Image shape: {images.shape}, Label shape: {labels.shape}, Class label shape: {y_cls.shape}, "
#                 f"Phase spectrum shape: {phase_image.shape}, Gradient spectrum shape: {gradient_image.shape}, HSV image shape: {hsv_image.shape}")
#             print(f"Label class range: {labels.min().item()} ~ {labels.max().item()}")
#             print(f"Label classes: {torch.unique(labels)}")
#             print(f"Classes: {y_cls}")
#             break
#
#         print("\nData loaded successfully! Ready for model training.")
#     except Exception as e:
#         print(f"\nData loading error: {str(e)}")

# if __name__ == "__main__":
#     config = {
#         "image_dir": "E:\\HE CONG BING _Downloads\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\JPEGImages",
#         "label_dir": "E:\\HE CONG BING _Downloads\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\SegmentationClass_Gray",
#         "train_split": "E:\\HE CONG BING _Downloads\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\ImageSets\\Segmentation\\train.txt",
#         "val_split": "E:\\HE CONG BING _Downloads\\PASCAL_VOC_2012\\VOC2012_train_val\\VOC2012_train_val\\ImageSets\\Segmentation\\val.txt",
#         "batch_size": 8,
#         "num_workers": 4,
#         "img_size": (512, 512)
#     }
#
#     print("Creating training dataloader...")
#     train_loader = create_segmentation_dataloader(
#         image_dir=config["image_dir"],
#         label_dir=config["label_dir"],
#         split_file=config["train_split"],
#         img_size=config["img_size"],
#         batch_size=config["batch_size"],
#         shuffle=True,
#         num_workers=config["num_workers"],
#         is_train=True
#     )
#
#     print("Creating validation dataloader...")
#     val_loader = create_segmentation_dataloader(
#         image_dir=config["image_dir"],
#         label_dir=config["label_dir"],
#         split_file=config["val_split"],
#         img_size=config["img_size"],
#         batch_size=config["batch_size"],
#         shuffle=False,
#         num_workers=config["num_workers"],
#         is_train=False
#     )
#
#     print(f"\nTraining set samples: {len(train_loader.dataset)}, batches: {len(train_loader)}")
#     print(f"Validation set samples: {len(val_loader.dataset)}, batches: {len(val_loader)}")
#
#     try:
#         print("\nTesting first batch of training set...")
#         for batch_idx, (images, labels) in enumerate(train_loader):
#             print(f"Training batch {batch_idx + 1} - Image shape: {images.shape}, Label shape: {labels.shape}")
#             print(f"Label class range: {labels.min().item()} ~ {labels.max().item()}")
#             print(f"Label classes: {torch.unique(labels)}")
#             break
#
#         print("\nTesting first batch of validation set...")
#         for batch_idx, (images, labels) in enumerate(val_loader):
#             print(f"Validation batch {batch_idx + 1} - Image shape: {images.shape}, Label shape: {labels.shape}")
#             print(f"Label class range: {labels.min().item()} ~ {labels.max().item()}")
#             break
#
#         print("\nData loaded successfully! Ready for model training.")
#     except Exception as e:
#         print(f"\nData loading error: {str(e)}")
