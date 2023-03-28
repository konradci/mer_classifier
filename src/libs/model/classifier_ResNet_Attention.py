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

########################################################################################################################
#                           protected under Request for grant of a European patent #P1349EP00                          #
########################################################################################################################

import torch
import torch.nn as nn

from libs.model.resnet import ResNetLayer


# resnet /w attention /w dropout
class Classifier_ResNet_Time_Attention_Drop(nn.Module):
    def __init__(self, dim_0f, dim_1t, bsize):
        super(Classifier_ResNet_Time_Attention_Drop, self).__init__()
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

        mid = 4000
        self.linmask1a = nn.Linear(128 * 65, mid)
        self.linmask1b = nn.Linear(mid, 128)
        self.linmask2a = nn.Linear(256 * 33, mid)
        self.linmask2b = nn.Linear(mid, 256)
        self.linmask3a = nn.Linear(512 * 17, mid)
        self.linmask3b = nn.Linear(mid, 512)
        self.linmask4a = nn.Linear(1024 * 9, mid)
        self.linmask4b = nn.Linear(mid, 1024)
        self.maskact = nn.Sigmoid()

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
        x = self.self_attention(x, self.linmask1a, self.linmask1b)
        x = self.resn2(x)
        x = self.self_attention(x, self.linmask2a, self.linmask2b)
        x = self.resn3(x)
        x = self.self_attention(x, self.linmask3a, self.linmask3b)
        x = self.resn4(x)
        x = self.self_attention(x, self.linmask4a, self.linmask4b)
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.decoder1(x)
        x = self.drop2(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x

    # self attention in temporal domain
    def self_attention(self, x, att_layer1, att_layer2):
        _bs, _ch, _fd, _td = [x.shape[j] for j in range(4)]
        x = x.permute((0, 3, 1, 2))
        x = x.view(_bs, _td, _ch * _fd)
        am = att_layer1(x)
        am = self.drop1(am)
        am = att_layer2(am)
        am = self.maskact(am)
        am = torch.unsqueeze(am, 3)
        x = x.view(_bs, _td, _ch, _fd)
        am = am.expand(x.shape)
        x = torch.mul(x, am)
        x = x.permute((0, 2, 3, 1))
        return x
