import torch
from torch import nn
import torchvision.transforms.functional as functional


class DoubleConvolution(nn.Module):
    """
    A module that performs a double convolution on a 2D input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, out_channels, H, W).

    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the DoubleConvolution module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        # First convolution layer with 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)

        # ReLU activation after first convolution
        self.act1 = nn.ReLU()

        # Second convolution layer with 3x3 kernel
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

        # ReLU activation after second convolution
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the DoubleConvolution module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, H, W).
        """
        # First convolution followed by ReLU activation
        x = self.conv1(x)
        x = self.act1(x)

        # Second convolution followed by ReLU activation
        x = self.conv2(x)
        x = self.act2(x)

        # Return the output tensor
        return x


class DownSample(nn.Module):
    """
    A module that performs max pooling on a 2D input tensor.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, in_channels, H/2, W/2).
    """

    def __init__(self) -> None:
        """
        Initializes the DownSample module.
        """
        super().__init__()

        # Max pooling layer with 2x2 kernel and stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the DownSample module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, in_channels, H/2, W/2).
        """
        # Apply max pooling and return the output tensor
        return self.maxpool(x)


class UpSample(nn.Module):
    """
    A module that performs transposed convolution on a 2D input tensor.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, out_channels, H*2, W*2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the UpSample module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()

        # Transposed convolution layer with 2x2 kernel and stride 2
        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the UpSample module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, H*2, W*2).
        """
        # Apply transposed convolution and return the output tensor
        return self.conv_trans(x)


class CropAndConcat(nn.Module):
    """
    A module that crops and concatenates two 2D input tensors along the channel axis.

    Args:
        x (torch.Tensor): The first input tensor.
        contracting_x (torch.Tensor): The second input tensor to be cropped and concatenated.

    Returns:
        torch.Tensor: The concatenated tensor of shape (batch_size, in_channels_x + in_channels_contracting_x, H, W).
    """

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the CropAndConcat module.

        Args:
            x (torch.Tensor): The first input tensor.
            contracting_x (torch.Tensor): The second input tensor to be cropped and concatenated.

        Returns:
            torch.Tensor: The concatenated tensor of shape (batch_size, in_channels_x + in_channels_contracting_x, H, W).
        """
        # Crop the second tensor to have the same size as the first tensor
        contracting_x = functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])

        # Concatenate the two tensors along the channel axis and return the output tensor
        x = torch.cat([x, contracting_x], dim=1)
        return x


class UNet(nn.Module):
    """
    A U-Net model for semantic segmentation of images.

    The architecture of the U-Net consists of a contracting path (downsampling path) and an expanding path (upsampling
    path). The contracting path consists of several blocks, each consisting of two 2D convolutions with ReLU activation
    and followed by 2D max pooling. The expanding path consists of several blocks, each consisting of an upsampling
    layer, concatenation with the corresponding block of the contracting path, followed by two 2D convolutions with
    ReLU activation. The final output of the network is a 2D convolutional layer with two output channels.

    Args:
        None

    Returns:
        torch.Tensor: The output tensor of shape (batch_size, 2, H, W).
    """

    def __init__(self) -> None:
        """
        Initializes the UNet module.

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        # Create the contracting path
        self.down_conv = nn.ModuleList(
            [
                DoubleConvolution(in_c, out_c)
                for in_c, out_c in [(1, 64), (64, 128), (128, 256), (256, 512)]
            ]
        )
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # Create the bottleneck layer
        self.bottleneck_conv = DoubleConvolution(512, 1024)

        # Create the expanding path
        self.up_sample = nn.ModuleList(
            [
                UpSample(in_c, out_c)
                for in_c, out_c in [(1024, 512), (512, 256), (256, 128), (128, 64)]
            ]
        )
        self.up_conv = nn.ModuleList(
            [
                DoubleConvolution(in_c, out_c)
                for in_c, out_c in [(1024, 512), (512, 256), (256, 128), (128, 64)]
            ]
        )
        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        # Create the final output layer
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass of the UNet module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, 1, H, W).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2, H, W).
        """
        features = []
        # Pass the input through the contracting path
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            features.append(x)
            x = self.down_sample[i](x)

        # Pass the input through the bottleneck layer
        x = self.bottleneck_conv(x)

        # Pass the input through the expanding path
        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x, features.pop())
            x = self.up_conv[i](x)

        # Pass the output through the final convolutional layer
        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    # Create a random tensor of size (1, 1, 572, 572)
    image = torch.rand((1, 1, 572, 572))
    # Initialize an instance of the UNet model
    model = UNet()
    # Pass the input tensor through the model
    result = model(image)
    # Print the size of the output tensor
    print(result.size())
