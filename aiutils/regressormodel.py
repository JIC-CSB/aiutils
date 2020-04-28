import torch.nn as nn
import torch.nn.functional as F


def down_block(in_ch, out_ch):

    padding = 1
    stride = 1
    kernel_size = 3

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    )


class RegressNet(nn.Module):

    model_desc = "Predict four floating point values per input image"
    model_name = "RegressNet"

    def __init__(self, input_channels=1):
        super().__init__()

        self.down0 = down_block(input_channels, 16) 
        self.down1 = down_block(16, 32)  # out 64x64
        self.down2 = down_block(32, 64)  # out 32x22
        self.down3 = down_block(64, 128)  # out 16x16
        self.down4 = down_block(128, 256)  # out 8x8
        self.down5 = down_block(256, 512)  # out 4x4
        self.fc1 = nn.Linear(4 * 4 * 4 * 512, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, xb):
        xb = self.down0(xb)
        xb = self.down1(xb)
        xb = self.down2(xb)
        xb = self.down3(xb)
        xb = self.down4(xb)
        xb = self.down5(xb)
        xb = xb.view(-1, 4 * 4 * 4 * 512)
        xb = F.relu(self.fc1(xb))
        xb = self.fc2(xb)
        return xb
