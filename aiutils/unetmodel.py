import dtoolcore

import numpy as np
import skimage.transform
import torch
from torch import nn
import torch.nn.functional as F

from torchvision.transforms.functional import to_tensor


def dice_coeff(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth)
            / (iflat.sum() + tflat.sum() + smooth))


def dice_loss(pred, target):

    return 1 - dice_coeff(pred, target)


def down_block(in_ch, out_ch):

    padding = 1
    stride = 1
    kernel_size = 3

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2, 2)
    )


class UNetUpsample(nn.Module):
    def __init__(self, scale_factor, mode):
        super(UNetUpsample, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x


def up_block(in_ch, out_ch):

    padding = 1
    stride = 1
    kernel_size = 3

    return nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        # F.interpolate(scale_factor=2, mode='bilinear', align_corners=True),
        UNetUpsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)

    )

class UNet(nn.Module):

    model_name = "SimpleUNet"

    def __init__(self, input_channels, output_channels=1, cl=True):
        super().__init__()
        cl = int(cl)
        self.down0 = down_block(input_channels, 32)
        self.down1 = down_block(32, 64)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.up4 = up_block(512, 256)
        self.up3 = up_block(256+cl*256, 128)
        self.up2 = up_block(128+cl*128, 64)
        self.up1 = up_block(64+cl*64, 32)
        self.up0 = up_block(32+cl*32, 16)

        self.cl = cl

        self.out = nn.Conv2d(16, output_channels, kernel_size=1)

    def forward(self, xb):
        xs = xb
        x0 = self.down0(xb)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        xb = self.up4(x4)
        if self.cl: xb = torch.cat([xb, x3], dim=1)

        xb = self.up3(xb)
        if self.cl: xb = torch.cat([xb, x2], dim=1)

        xb = self.up2(xb)
        if self.cl: xb = torch.cat([xb, x1], dim=1)

        xb = self.up1(xb)
        if self.cl: xb = torch.cat([xb, x0], dim=1)

        xb = self.up0(xb)
        # xb = torch.cat([xb, xs], dim=1)

        xb = self.out(xb)

        return torch.sigmoid(xb)


def unet_model_from_uri(uri, input_channels=1):
    model = UNet(input_channels=input_channels)

    model_idn = dtoolcore.utils.generate_identifier("model.pt")
    ds = dtoolcore.DataSet.from_uri(uri)
    state_abspath = ds.item_content_abspath(model_idn)

    model.load_state_dict(torch.load(state_abspath, map_location='cpu'))

    model.eval()

    return model


class TrainedUNet(object):

    def __init__(self, uri, input_channels=1):
        self.model = unet_model_from_uri(uri, input_channels=input_channels)

    def predict_mask_from_tensor(self, input_tensor):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_tensor = input_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            pred_mask_tensor = self.model(input_tensor[None])

        return pred_mask_tensor.squeeze().cpu().numpy()

    def predict_mask_from_image(self, im):
        input_tensor = to_tensor(im)
        return self.predict_mask_from_tensor(input_tensor)

    def scaled_mask_from_image(self, im, dim=(512, 512)):
        scaled_im = skimage.transform.resize(im, dim).astype(np.float32)

        scaled_mask = self.predict_mask_from_image(scaled_im)

        rdim, cdim, _ = im.shape
        mask = skimage.transform.resize(scaled_mask, (rdim, cdim))

        return mask


