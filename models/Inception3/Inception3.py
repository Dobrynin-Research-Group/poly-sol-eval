from torch import Tensor, cat
from torch.nn import (
    AdaptiveAvgPool2d,
    AvgPool2d,
    Conv2d,
    Flatten,
    functional,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
)


class BasicConv2d(Module):
    """Custom Conv2d Module with :code:`bias=False` by default."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__()

        self.conv = Conv2d(in_channels, out_channels, bias=False, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return functional.relu(x, inplace=True)


class Conv2dTo2Channels(Module):
    """Custom Module that creates two channels, one each from a pass of a
    :class:`BasicConv2d` with orthogonal, asymmetric kernels. Equivalent to:

    Example:
        This code:
        >>> x = torch.rand((8, 8))
        >>> model = Conv2dTo2Channels(4, 4, kernel_size=(1, 3), padding=(0, 1))
        >>> y = model(x)

        should be equivalent to:
        >>> x = torch.rand((8, 8))
        >>> model1 = BasicConv2d(4, 4, kernel_size=(1, 3), padding=(0, 1))
        >>> model2 = BasicConv2d(4, 4, kernel_size=(3, 1), padding=(1, 0))
        >>> y = torch.cat((model1(x), model2(x)), 1)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        padding: tuple[int, int],
    ):
        super().__init__()

        self.channel1 = BasicConv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.channel2 = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size[1], kernel_size[0]),
            padding=(padding[1], padding[0]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cat((self.channel1(x), self.channel2(x)), 1)


class InceptionA(Module):
    """An inception block containing four branches:
    1. kernel_size=1
    2. kernel_size=1 -> kernel_size=5
    3. kernel_size=1 -> kernel_size=3 -> kernel_size=3
    4. AvgPool2d(kernel_size=3) -> kernel_size=1
    """

    def __init__(self, in_channels: int, pool_features: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5 = Sequential(
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2),
        )
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cat(
            (
                self.branch1x1(x),
                self.branch5x5(x),
                self.branch3x3dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class InceptionB(Module):
    """An inception block containing three branches:
    1. kernel_size=3
    2. kernel_size=1 -> kernel_size=3 -> kernel_size=3
    3. AvgPool2d(kernel_size=3)
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2),
        )
        self.branch_pool = AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor):
        return cat((self.branch3x3(x), self.branch3x3dbl(x), self.branch_pool(x)), 1)


class InceptionC(Module):
    """An inception block containing four branches:
    1. kernel_size=1
    2. kernel_size=1 -> kernel_size=(1, 7) -> kernel_size=(7, 1)
    3. kernel_size=1 -> kernel_size=(7, 1) -> kernel_size=(1, 7)
       -> kernel_size=(7, 1) -> kernel_size=(1, 7)
    3. kernel_size=1 -> kernel_size=3 -> kernel_size=3
    4. AvgPool2d(kernel_size=3) -> kernel_size=1
    """

    def __init__(self, in_channels: int, channels_7x7: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7 = Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch7x7dbl = Sequential(
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cat(
            (
                self.branch1x1(x),
                self.branch7x7(x),
                self.branch7x7dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class InceptionD(Module):
    """An inception block containing three branches:
    1. kernel_size=1 -> kernel_size=3
    2. kernel_size=1 -> kernel_size=(1, 7) -> kernel_size=(7, 1)
       -> kernel_size=3
    3. AvgPool2d(kernel_size=3)
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.branch3x3 = Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2),
        )
        self.branch7x7x3 = Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch_pool = AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        return cat((self.branch3x3(x), self.branch7x7x3(x), self.branch_pool(x)), 1)


class InceptionE(Module):
    """An inception block containing four branches:
    1. kernel_size=1
    2. kernel_size=1 -> Conv2dTo2Channels(kernel_size=(1, 3))
    3. kernel_size=1 -> kernel_size=3 -> Conv2dTo2Channels(kernel_size=(1, 3))
    4. AvgPool2d(kernel_size=3) -> kernel_size=1
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3 = Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            Conv2dTo2Channels(384, 384, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch3x3dbl = Sequential(
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3, padding=1),
            Conv2dTo2Channels(384, 384, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch_pool = Sequential(
            AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return cat(
            (
                self.branch1x1(x),
                self.branch3x3(x),
                self.branch3x3dbl(x),
                self.branch_pool(x),
            ),
            1,
        )


class Inception3(Module):
    """Inception-block-based neural network for training on 2D images."""

    def __init__(self):
        super().__init__()

        self.stack = Sequential(
            BasicConv2d(1, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
            MaxPool2d(kernel_size=3, stride=2),
            InceptionA(192, pool_features=32),
            InceptionA(256, pool_features=64),
            InceptionA(288, pool_features=64),
            InceptionB(288),
            InceptionC(768, channels_7x7=128),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=160),
            InceptionC(768, channels_7x7=192),
            InceptionD(768),
            InceptionE(1280),
            InceptionE(2048),
            AdaptiveAvgPool2d((1, 1)),
            Flatten(1),
            Linear(2048, 1),
            Flatten(0),
        )

    def _transform_input(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.view(x.shape[0], 1, *x.shape[1:])
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._transform_input(x)
        return self.stack(x)
