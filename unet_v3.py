import torch
from torch import nn
from torchvision.transforms import functional


class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        return x


class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.maxpool(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.conv_trans(x)


class CropAndConcat(nn.Module):
    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        contracting_x = functional.center_crop(
            contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConvolution(in_c, out_c) for in_c, out_c in [
                                       (1, 64), (64, 128), (128, 256), (256, 512)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.bottleneck_conv = DoubleConvolution(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(in_c, out_c) for in_c, out_c in [
                                       (1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConvolution(in_c, out_c) for in_c, out_c in [
                                     (1024, 512), (512, 256), (256, 128), (128, 64)]])
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = []
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            features.append(x)
            x = self.down_sample[i](x)

        x = self.bottleneck_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, features.pop())
            x = self.up_conv[i](x)

        x = self.final_conv(x)
        return x


if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    result = model(image)
    print(result.size())
