from torch import Tensor
from torch.nn import LazyConv2d, MaxPool2d, Module


class AlexNetBackbone(Module):
    """Tracker backbone implemented using initial AlexNet
    backbone architecture with little adjustments.

    Notes
    -----
    For more info proceed to the reading of the paper
    "Fully-Convolutional Siamese Networks for Object Tracking"
    (https://arxiv.org/pdf/1606.09549).

    """

    def __init__(self) -> None:
        super(AlexNetBackbone, self).__init__()

        self.conv1 = LazyConv2d(out_channels=96, kernel_size=(11, 11), stride=2)
        self.pool1 = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv2 = LazyConv2d(out_channels=256, kernel_size=(5, 5), stride=1)
        self.pool2 = MaxPool2d(kernel_size=(3, 3), stride=2)
        self.conv3 = LazyConv2d(out_channels=192, kernel_size=(3, 3), stride=1)
        self.conv4 = LazyConv2d(out_channels=192, kernel_size=(3, 3), stride=1)
        self.conv5 = LazyConv2d(out_channels=128, kernel_size=(3, 3), stride=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)
