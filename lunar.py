import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# ----------------------------------------------------------------------------------
# ClassificationModel (Chandrayaan-2 Ready: Metadata-Aware)
# ----------------------------------------------------------------------------------
# Context:
# This model framework was provided as a template by a teammate, with instructions
# to modify key architectural elements based on dataset properties like:
#   - Number of input channels (e.g. OHRC + slope + sun elevation)
#   - Kernel sizes, padding, or model depth (based on tile resolution)
#
# Dataset Handling:
# Chandrayaan-2 OHRC images are massive (~50GB per frame). Each image is sliced
# into 224x224 patches, each paired with annotations and metadata for:
#   - Latitude, longitude (from .oat)
#   - Sun azimuth & elevation (from .spm)
#   - Solar incidence angle (derived from elevation)
#   - Slope/Digital Terrain Model (from DTM raster if used)
# Work Developed by ml_agg & 51riu5
# Model Design:
# - Accepts stacked multi-channel image tiles (e.g., grayscale + sun elevation + slope)
# - Binary classification: 1 = boulder/landslide, 0 = none
# - Output = sigmoid(logits) for per-tile probability
# ----------------------------------------------------------------------------------

class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, bias=False, groups=1):
        super().__init__()
        padding = ks // 2
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MultiScaleConv(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.conv1x1 = ConvRelu(dim, dim, ks=1, bias=bias)
        self.conv3x3 = ConvRelu(dim, dim, ks=3, bias=bias, groups=dim)
        self.conv5x5 = ConvRelu(dim, dim, ks=5, bias=bias, groups=dim)
        self.conv7x7 = ConvRelu(dim, dim, ks=7, bias=bias, groups=dim)
        self.conv9x9 = ConvRelu(dim, dim, ks=9, bias=bias, groups=dim)
        self.fuse = ConvRelu(5 * dim, dim, ks=1, bias=bias)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        x7 = self.conv7x7(x)
        x9 = self.conv9x9(x)
        return self.fuse(torch.cat([x1, x3, x5, x7, x9], dim=1))


class MultiScaleConvNxt(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale=1e-6, bias=False):
        super().__init__()
        self.conv = MultiScaleConv(dim, bias=bias)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True) if layer_scale > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return res + self.drop_path(x)


class ClassificationModel(nn.Module):
    def __init__(self, in_channel=3, num_classes=1, channels=[16, 32, 64, 96, 128], num_layers=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.blocks = nn.Sequential(
            self._make_block(in_channel, channels[0], num_layers),
            self._make_block(channels[0], channels[1], num_layers),
            self._make_block(channels[1], channels[2], num_layers),
            self._make_block(channels[2], channels[3], num_layers),
            self._make_block(channels[3], channels[4], num_layers)
        )
        self.fc = nn.Linear(channels[-1], num_classes)

    def _make_block(self, in_ch, out_ch, num_layers):
        layers = [ConvRelu(in_ch, out_ch)]
        layers += [MultiScaleConvNxt(out_ch) for _ in range(num_layers)]
        return nn.Sequential(*layers, self.pool)

    def forward(self, x):
        x = self.blocks(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    # Simulate lunar tile input (3 channels = image + sun elev + incidence)
    model = ClassificationModel(in_channel=3, num_classes=1)

    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)
    print("Model Output Shape:", output.shape)
