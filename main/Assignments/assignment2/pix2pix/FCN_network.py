import torch
import torch.nn as nn

class FullyConvNetwork(nn.Module):
    def __init__(self):
        super(FullyConvNetwork, self).__init__()

        # 编码部分（下采样）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 增大通道数
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 增大通道数
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 增大通道数
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # 增大通道数
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)  # 增大通道数
        self.bn5 = nn.BatchNorm2d(1024)

        # 解码部分（上采样 + 逐步细化模块）
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 保持对应的通道数
        self.refine1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # 增大通道数
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 保持对应的通道数
        self.refine2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 增大通道数
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 保持对应的通道数
        self.refine3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 增大通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 保持对应的通道数
        self.refine4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 增大通道数
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 输出层
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)  # 保持输出通道为3

    def forward(self, x):
        # 编码部分
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        x4 = self.relu(self.bn4(self.conv4(x3)))
        x5 = self.relu(self.bn5(self.conv5(x4)))

        # 解码部分，添加跳跃连接和逐步细化模块
        x = self.upconv1(x5)
        x = self.refine1(torch.cat([x, x4], dim=1))  # x 应该与 x4 通道数拼接

        x = self.upconv2(x)
        x = self.refine2(torch.cat([x, x3], dim=1))  # x 应该与 x3 通道数拼接

        x = self.upconv3(x)
        x = self.refine3(torch.cat([x, x2], dim=1))  # x 应该与 x2 通道数拼接
        
        x = self.upconv4(x)
        x = self.refine4(torch.cat([x, x1], dim=1))  # x 应该与 x1 通道数拼接

        # 输出
        x = self.output_conv(x)
        return x