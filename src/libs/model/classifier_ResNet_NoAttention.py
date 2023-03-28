# Copyright (C) 2023  NASK PIB
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch
import torch.nn as nn

from libs.model.resnet import ResNetLayer


# resnet /wo attention
class Classifier_ResNet_NoAttention(nn.Module):
    def __init__(self, dim_0f, dim_1t, bsize):
        super(Classifier_ResNet_NoAttention, self).__init__()
        self.dim_0f = dim_0f
        self.dim_1t = dim_1t

        if isinstance(bsize, int):
            bsize = [bsize for _ in range(4)]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.resn1 = ResNetLayer(in_channels=64, out_channels=128, n=bsize[0])
        self.resn2 = ResNetLayer(in_channels=128, out_channels=256, n=bsize[1])
        self.resn3 = ResNetLayer(in_channels=256, out_channels=512, n=bsize[2])
        self.resn4 = ResNetLayer(in_channels=512, out_channels=1024, n=bsize[3])

        self.avg = nn.AdaptiveAvgPool2d((6, 6))
        self.decoder1 = nn.Linear(36 * 1024, 8192)
        self.decoder2 = nn.Linear(8192, 4096)
        self.decoder3 = nn.Linear(4096, 1)

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        x = torch.unsqueeze(x, 1)
        x = self.head(x)
        x = self.resn1(x)
        x = self.resn2(x)
        x = self.resn3(x)
        x = self.resn4(x)

        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.decoder1(x)
        x = self.drop2(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x
