import torch
import pretrainedmodels
import torch.nn as nn

# Load pre-trained Xception
xception = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')

# Remove classification layers
modules = list(xception.children())[:-2]  # Remove final fully connected layer and global pooling layer
xception_backbone = nn.Sequential(*modules)

# Print model structure for confirmation
print(xception_backbone)

# Test input
input_image = torch.randn(1, 3, 299, 299)  # Assume input is 299x299 RGB image
output = xception_backbone(input_image)
print(output.shape)  # Output shape should be (1, 2048, 9, 9), 2048 channels with 9x9 spatial size
