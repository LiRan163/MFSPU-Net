import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional
import extractor


class SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling module.
    Two modes:
      - mode='flatten': classic SPP -> outputs a flattened vector of size C * sum(p*p for p in pool_sizes)
      - mode='spatial': pyramid pooling for segmentation/detection -> outputs tensor with channels C*(1+len(pool_sizes))
                       (optionally passed through a 1x1 conv bottleneck to reduce channels)
    Args:
        pool_sizes: iterable of ints for pyramid levels (e.g. (1, 2, 3, 6))
        mode: 'flatten' or 'spatial'
        in_channels: required if mode=='spatial' and out_channels specified (for bottleneck conv)
        out_channels: if not None and mode=='spatial', apply 1x1 conv to reduce channels to out_channels
        pool_type: 'avg' or 'max' pooling (uses adaptive pooling)
    """

    def __init__(
            self,
            pool_sizes: Iterable[int] = (1, 2, 3, 6),
            mode: str = "spatial",
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            pool_type: str = "avg",
    ):
        super().__init__()
        assert mode in ("flatten", "spatial"), "mode must be 'flatten' or 'spatial'"
        assert pool_type in ("avg", "max")
        self.pool_sizes = tuple(pool_sizes)
        self.mode = mode
        self.pool_type = pool_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.mode == "spatial" and out_channels is not None:
            if in_channels is None:
                raise ValueError("in_channels must be provided when out_channels is set for spatial mode")
            self.bottleneck = nn.Sequential(
                nn.Conv2d(in_channels * (1 + len(self.pool_sizes)), out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.bottleneck = None

    def _adaptive_pool(self, x: torch.Tensor, output_size: int) -> torch.Tensor:
        if self.pool_type == "avg":
            return F.adaptive_avg_pool2d(x, output_size)
        else:
            return F.adaptive_max_pool2d(x, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape

        if self.mode == "flatten":
            pooled_vectors = []
            for p in self.pool_sizes:
                out = self._adaptive_pool(x, (p, p))  # (N, C, p, p)
                pooled_vectors.append(out.view(N, -1))  # flatten
            out = torch.cat(pooled_vectors, dim=1)
            return out

        pooled_maps = [x]  # include original feature map
        for p in self.pool_sizes:
            out = self._adaptive_pool(x, (p, p))
            if (p, p) != (H, W):
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            pooled_maps.append(out)
        out = torch.cat(pooled_maps, dim=1)  # concat along channels

        if self.bottleneck is not None:
            out = self.bottleneck(out)
        return out


class DoubleConv(nn.Module):
    """(Convolution => BN => ReLU) * 2, supports dilated convolutions (dilation)"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1, num_groups=32):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            mid_channels: Intermediate convolution channels, defaults to out_channels if None
            dilation: Dilation rate, default=1
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False
            ),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ChannelAttentionFusion(nn.Module):

    def __init__(self, input_size, pool_size, hidden_size):
        super(ChannelAttentionFusion, self).__init__()
        self.input_size = input_size
        self.pool_size = pool_size
        self.GAP = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.GMP = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.w1 = nn.Linear(in_features=pool_size * pool_size, out_features=1)
        self.w2 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(
            self.w1(torch.add(self.GMP(x), self.GAP(x)).reshape(x.shape[0], x.shape[1], -1)).unsqueeze(dim=3))
        return self.w2(torch.add(x * att, x))


class Down(nn.Module):
    """Downsampling module: MaxPool => DoubleConv"""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation),
            SpatialPyramidPooling(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling module"""

    def __init__(self, in_channels, out_channels, dilation, bilinear=True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels, in_channels // 2, dilation=dilation),
                SpatialPyramidPooling(in_channels=out_channels, out_channels=out_channels)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                DoubleConv(in_channels, out_channels, dilation=dilation),
                SpatialPyramidPooling(in_channels=out_channels, out_channels=out_channels)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Output convolution layer, maps features to class count"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net main model"""

    def __init__(self, n_channels, n_classes, bilinear=True, backend='resnet101', pretrained=True, if_pre=False,
                 if_class=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.if_pre = if_pre
        self.if_class = if_class
        self.down_dilation = (1, 2, 4, 8)
        self.up_dilation = (8, 4, 2, 1)

        # self.inc = DoubleConv(n_channels, 64)
        if self.if_pre:
            self.inc = getattr(extractor, backend)(pretrained)
            self.conv1 = nn.Sequential(
                DoubleConv(2048, 64, 256),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
            self.conv2 = nn.Sequential(
                DoubleConv(2048, 64, 256),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
            self.conv3 = nn.Sequential(
                DoubleConv(2048, 64, 256),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
            self.conv4 = nn.Sequential(
                DoubleConv(2048, 64, 256),
                nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            )
            self.conv_cat = ChannelAttentionFusion(input_size=64 * 4, pool_size=64, hidden_size=64)
        else:
            self.conv_cat = ChannelAttentionFusion(input_size=self.n_channels * 4, pool_size=64, hidden_size=64)
        self.down1 = Down(64, 128, self.down_dilation[0])
        self.down2 = Down(128, 256, self.down_dilation[1])
        self.down3 = Down(256, 512, self.down_dilation[2])
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, self.down_dilation[3])

        self.up1 = Up(1024, 512 // factor, self.up_dilation[0], bilinear)
        self.up2 = Up(512, 256 // factor, self.up_dilation[1], bilinear)
        self.up3 = Up(256, 128 // factor, self.up_dilation[2], bilinear)
        self.up4 = Up(128, 64, self.up_dilation[3], bilinear)

        self.outc = OutConv(64, n_classes)

        if self.if_class and self.if_pre:
            self.classifier = nn.Sequential(
                nn.Linear(1024 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )
        elif self.if_class and not self.if_pre:
            self.classifier = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )

    def forward(self, x, x_phase, x_gradient, x_hsv):
        if self.if_pre:
            x_cat = self.conv_cat(torch.cat(
                [self.conv1(self.inc(x)[0]), self.conv2(self.inc(x_phase)[0]), self.conv3(self.inc(x_gradient)[0]),
                 self.conv4(self.inc(x_hsv)[0])], dim=1))
            x_cat = F.interpolate(x_cat, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            auxiliary = F.adaptive_max_pool2d(
                input=torch.cat([self.inc(x)[1], self.inc(x_phase)[1], self.inc(x_gradient)[1], self.inc(x_hsv)[1]],
                                dim=1),
                output_size=(1, 1)).view(-1, 1024 * 4)
        else:
            x_cat = self.conv_cat(torch.cat([x, x_phase, x_gradient, x_hsv], dim=1))
            auxiliary = F.adaptive_max_pool2d(input=x_cat, output_size=(1, 1)).view(-1, 64)
        x2 = self.down1(x_cat)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x_cat)
        if self.if_class:
            return self.outc(x), self.classifier(auxiliary)
        else:
            return self.outc(x)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
# if __name__ == "__main__":
#     model = UNet(n_channels=3, n_classes=21, bilinear=True, if_pre=True, if_class=False)
#     print(model)
#     print('The total number of parameters', count_parameters(model))
#     input_1 = torch.randn(1, 3, 428, 510)
#     input_2 = torch.randn(1, 3, 428, 510)
#     input_3 = torch.randn(1, 3, 428, 510)
#     input_4 = torch.randn(1, 3, 428, 510)
#     out = model(input_1, input_2, input_3, input_4)
#     print(out.shape)
