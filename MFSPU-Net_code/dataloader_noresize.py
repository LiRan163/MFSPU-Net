import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, split_file,
                 transform_img=None, transform_label=None):

        self.image_dir = image_dir
        self.label_dir = label_dir

        with open(split_file, 'r') as f:
            self.sample_names = [line.strip() for line in f if line.strip()]

        self.transform_img = transform_img if transform_img is not None else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_label = transform_label if transform_label is not None else transforms.Compose([
            transforms.ToTensor()
        ])

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

        image = self.transform_img(image)
        label = self.transform_label(label)

        label = (label * 255).long().squeeze()

        assert image.shape[1:] == label.shape, \
            f"Image size {image.shape[1:]} does not match label size {label.shape}: {sample_name}"

        return image, label


def create_segmentation_dataloader(
        image_dir,
        label_dir,
        split_file,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        is_train=True
):
    if is_train:
        transform_img = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.2),
            # transforms.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        transform_label = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.2),
            # transforms.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    else:
        transform_img = None
        transform_label = None

    dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        split_file=split_file,
        transform_img=transform_img,
        transform_label=transform_label
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