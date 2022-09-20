import torch
import torch.nn as nn


def double_conv(input_channels, output_channels):
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


def crop_img(tensor, target_tensor):
    tensor_size = tensor.shape[2]
    target_size = target_tensor.shape[2]
    delta = tensor_size - target_size
    delta = delta//2
    return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]


class UNet(nn.Module):
    def __init__(self) -> None:
        super(UNet, self).__init__()

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(double_conv(1, 64)),
            nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                          double_conv(64, 128)),
            nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                          double_conv(128, 256)),
            nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                          double_conv(256, 512)),
            nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                          double_conv(512, 1024)),
        ])

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(
                1024, 512, kernel_size=2, stride=2)),
            nn.Sequential(double_conv(1024, 512),
                          nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)),
            nn.Sequential(double_conv(512, 256),
                          nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)),
            nn.Sequential(double_conv(256, 128),
                          nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)),
        ])

        self.output_block = nn.Sequential(nn.Sequential(double_conv(128, 64)),
                                          nn.Conv2d(64, 2, kernel_size=1))

    def forward(self, image):
        feats = []
        x = image
        for f in self.encoder_blocks:
            x = f(x)
            feats.append(x)

        x = feats.pop()
        for f in self.decoder_blocks:
            x = f(x)
            try:
                x = torch.concat((x, crop_img(feats[-1], x)), 1)
            except Exception as err:
                print(x.size())
                print(feats[-1].size())
                raise err
            feats.pop()

        x = self.output_block(x)
        return x


if __name__ == '__main__':
    image = torch.rand((1, 1, 572, 572))
    model = UNet()
    result = model(image)
    print(result.size())
