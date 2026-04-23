# MFSPU-Net
# MFSPU-Net

A multi-input U-Net style semantic segmentation project that fuses RGB images with auxiliary phase, gradient, and HSV representations.

## Overview

MFSPU-Net is a semantic segmentation framework built on top of a U-Net-like encoder-decoder architecture.  
The project is designed to combine multiple visual representations of the same image, including:

- RGB image
- Phase image
- Gradient image
- HSV image

These inputs are fused before the main segmentation backbone and then passed through a U-Net-style network with dilated convolutions, spatial pyramid pooling, and attention-based feature fusion.

At the current stage, the training and testing scripts are configured for **PASCAL VOC 2012** style semantic segmentation experiments.

---

## Main Features

- Multi-input feature fusion for semantic segmentation
- U-Net style encoder-decoder architecture
- Channel-attention-based fusion module
- Dilated convolutions in encoder and decoder blocks
- Spatial pyramid pooling modules
- Supports optional pretrained backbone extractor
- Training, validation, visualization, and metric evaluation scripts
- Utilities for label conversion and auxiliary preprocessing experiments

---

## Repository Structure

```text
MFSPU-Net/
├── README.md
└── MFSPU-Net_code/
    ├── train.py
    ├── test.py
    ├── unet.py
    ├── dataloader.py
    ├── dataloader_noresize.py
    ├── metric.py
    ├── extractor.py
    ├── xception.py
    ├── Cityscapes_Transformer.py
    ├── Gabor_k-means.py
    ├── Otsu.py
    └── contrastive_model_test_voc.py
