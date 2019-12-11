from __future__ import print_function

import torch
import torch.nn as nn
import blend

class VGG16Net(nn.Module):

    def __init__(self):
        super(VGG16Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.features(x)

    def load_pretrained(self, pretrained_pth):
        dst_state = self.state_dict()
        src_state = torch.load(pretrained_pth)
        # print(type(dst_state.values()[0]))
        # print(type(src_state.values()[0]))
        for dst_param, src_param in zip(dst_state.values(), src_state.values()):
            dst_param.copy_(src_param)



class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        h = self.residual(x)
        return h + x


class multiDeblendNet(nn.Module):
    def __init__(self):
        super(multiDeblendNet, self).__init__()

        # 4 x 256 x 256
        # 32 x 256 x 256
        self.conv0 = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=5 , stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(66, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 256 x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(130, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 512 x 32 x 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # 256 x 64 x 64
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(130, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 32 x 256 x 256
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(66, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 3 x 256 x 256
        self.conv = nn.Sequential(
            nn.Conv2d(34, 7, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x, edge_x1, edge_x2):
        e1 = edge_x1
        e2 = edge_x2
        x0 = torch.cat((x, e1), 1)
        x0 = torch.cat((x0, e2), 1)
        h0 = self.conv0(x0)
        h1 = self.conv1(h0)
        e3 = blend.Down_H_map(edge_x1, 2)
        e4 = blend.Down_H_map(edge_x2, 2)
        h1 = torch.cat((h1, e3), 1)
        h1 = torch.cat((h1, e4), 1)
        h2 = self.conv2(h1)
        e5 = blend.Down_H_map(edge_x1, 4)
        e6 = blend.Down_H_map(edge_x2, 4)
        h2 = torch.cat((h2, e5), 1)
        h2 = torch.cat((h2, e6), 1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        h_res = self.res(h4)

        h = self.deconv4(h_res)
        h = self.deconv3(h)
        h = torch.cat((h, e5), 1)
        h = torch.cat((h, e6), 1)
        h = self.deconv2(h)
        h = torch.cat((h, e3), 1)
        h = torch.cat((h, e4), 1)
        h = self.deconv1(h)
        h = torch.cat((h, e1), 1)
        h = torch.cat((h, e2), 1)
        h = self.conv(h)

        y1 = h[:, 0:3, :, :]
        y2 = h[:, 3:6, :, :]
        mask = h[:, 6:7, :, :]
        # y1, y2 = h.chunk(2, dim=1)
        return y1, y2 , mask



class InpNet(nn.Module):

    def __init__(self):
        super(InpNet, self).__init__()
        # 4 x 256 x 256
        # 32 x 256 x 256
        self.conv0 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5 , stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 256 x 32 x 32
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 512 x 32 x 32
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )
        # 256 x 64 x 64
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 128 x 64 x 64
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # 64 x 128 x 128
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 32 x 256 x 256
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 3 x 256 x 256
        self.conv = nn.Sequential(
            # nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2),
            # nn.Sigmoid(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid(),
        )


    def forward(self, x):
        h0 = self.conv0(x)
        h1 = self.conv1(h0)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        h_res = self.res(h4)

        h = self.deconv4(h_res)
        h = torch.cat([h, h3], 1)
        h = self.deconv3(h)
        h = torch.cat([h, h2], 1)
        h = self.deconv2(h)
        h = torch.cat([h, h1], 1)
        h = self.deconv1(h)
        h = torch.cat([h, h0], 1)
        y = self.conv(h)

        return y


class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()

        self.critic = nn.Sequential(
            # 3 x 256 x 256
            nn.Conv2d(3, 32, kernel_size=6 , stride=2, padding=2),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            # 32 x 128 x 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            # 128 x 32 x 32
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            # 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            # 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512 x 8 x 8
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            # 512 x 4 x 4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
            # 1 x 1 x 1
        )

    def forward(self, x):
        return self.critic(x)

