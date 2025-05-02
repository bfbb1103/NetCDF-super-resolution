from torch import nn
import math

#实现自定义的卷积模块
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm=False, activation=None):
        super().__init__()
        self.layers = []

        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     bias=False, padding=kernel_size // 2))
        if batch_norm:
            self.layers.append(nn.BatchNorm2d(num_features=out_channels))
        if activation is not None:
            if activation == 'prelu':
                self.layers.append(nn.PReLU())
            if activation == 'Tanh':
                self.layers.append(nn.Tanh())
            if activation == 'relu':
                self.layers.append(nn.ReLU(inplace=True))

    def __call__(self, x):
        _identity = x
        output = []
        # 向前传播
        for layer in self.layers:
            output = layer(x)
            x = output
        assert output is not None, "output is None"
        return output

#实现自定义的残差模块
class ResidualBlock(nn.Module):
    expansion = 1 #通道拓展倍数
    """
        kernel_size 输入大小
        n_channels 输出大小
        stride_ 步长
    """
    def __init__(self, kernel_size, n_channels, stride_ = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, bias=False,padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3,
                               stride=stride_, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def __call__(self, x):
        identity = x

        # 第一阶段
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        # 第二阶段（主路径）
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += identity
        out = self.prelu(out)

        return out

# 子像素卷积模块
class SubPixelConvolutionalBlock(nn.Module):
    def __init__(self, kernel_size, n_channels, scaling_factor=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2), kernel_size=kernel_size, padding=kernel_size // 2)
        self.pixelshuffle1 = nn.PixelShuffle(scaling_factor)
        # self.pixelshuffle2 = nn.PixelShuffle(scaling_factor)
        self.prelu = nn.PReLU()


    def __call__(self, x):
        out = self.conv1(x)
        out = self.pixelshuffle1(out)
        # out = self.pixelshuffle2(out)
        out = self.prelu(out)

        return out

# SRResNet模型
class SRresNet(nn.Module):
    ## -----------------------------------------------
    # large_kernel_size: 第一层卷积和最后一层卷积核大小
    # small_kernel_size: 卷积核的大小
    # n_channels: 中间通道数
    # n_blocks: 残差模块数
    # scaling_factor： 缩放比例
    ## -----------------------------------------------
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=0.25):
        super(SRresNet, self).__init__()
        self.name = "SRresNet"
        scaling_factor = float(scaling_factor)
        assert scaling_factor in {0.5, 0.25, 0.125}, "缩放比例必须为0.5，0.25，0.125"

        #第一个卷积模块, 灰度图像 in_channels 值为1
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=large_kernel_size)

        # 一系列残差模块, 每个残差模块包含一个跳连接
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(kernel_size=small_kernel_size, n_channels=n_channels, stride_=1) for i in range(n_blocks)])

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(1/scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=1, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_tensor):
        """
        前向传播.
        :参数 lr_tensor: 低分辨率输入图像集, 张量表示，大小为 (N, 1, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 1, w * scaling factor, h * scaling factor)
        """
        output = self.conv_block1(lr_tensor)                    # (8, 1, 12, 12)
        residual = output                                       # (8, 64, 12, 12)
        output = self.residual_blocks(output)                   # (8, 64, 12, 12)
        output = self.conv_block2(output)                       # (8, 64, 12, 12)
        output = output + residual                              # (8, 64, 12, 12)
        output = self.subpixel_convolutional_blocks(output)     # (8, 64, 12 * 4, 12 * 4)
        sr_tensor = self.conv_block3(output)                    # (8, 1, 12 * 4, 12 * 4)

        return sr_tensor

# CNN模型
class SRCNN(nn.Module):
    ## -----------------------------------------------
    # large_kernel_size: 第一层卷积和最后一层卷积核大小
    # small_kernel_size: 卷积核的大小
    # n_channels: 中间通道数
    # scaling_factor： 缩放比例
    ## -----------------------------------------------
    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=0.25):
        super(SRCNN, self).__init__()
        self.name = "SRCNN"
        scaling_factor = float(scaling_factor)
        assert scaling_factor in {0.5, 0.25, 0.125}, "缩放比例必须为0.5，0.25，0.125"

        #第一个卷积模块, 灰度图像 in_channels 值为1
        self.conv_block1 = ConvolutionalBlock(in_channels=1, out_channels=n_channels, kernel_size=large_kernel_size)

        # 第二个卷积块
        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        # 放大通过子像素卷积模块实现, 每个模块放大两倍
        n_subpixel_convolution_blocks = int(math.log2(1/scaling_factor))
        self.subpixel_convolutional_blocks = nn.Sequential(
            *[SubPixelConvolutionalBlock(kernel_size=small_kernel_size, n_channels=n_channels, scaling_factor=2) for i
              in range(n_subpixel_convolution_blocks)])

        # 最后一个卷积模块
        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=1, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_tensor):
        """
        前向传播.
        :参数 lr_tensor: 低分辨率输入图像集, 张量表示，大小为 (N, 1, w, h)
        :返回: 高分辨率输出图像集, 张量表示， 大小为 (N, 1, w * 1/scaling factor, h * 1/scaling factor)
        """
        output = self.conv_block1(lr_tensor)                    # (8, 1, 12, 12)
        output = self.conv_block2(output)                       # (8, 64, 12, 12)
        output = self.subpixel_convolutional_blocks(output)     # (8, 64, 12 * 4, 12 * 4)
        sr_tensor = self.conv_block3(output)                    # (8, 1, 12 * 4, 12 * 4)

        return sr_tensor    ## -----------------------------------------------

