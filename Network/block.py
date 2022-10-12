import paddle
import paddle.nn as nn


class Conv(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size, stride=1, groups=1, bn=False, act=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2D(in_c, out_c, kernel_size, stride, kernel_size // 2, groups=groups)
        self.bn = nn.BatchNorm2D(out_c) if bn else None
        self.act = act if act else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class DeConv(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=1, group=1, bn=False, act=None):
        super(DeConv, self).__init__()
        self.deconv = nn.Conv2DTranspose(in_c, out_c, kernel_size, stride, groups=group, padding=padding)
        self.bn = nn.BatchNorm2D(out_c) if bn else None
        self.act = act if act else None

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class SEBlock(nn.Layer):
    def __init__(self, channels, t):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.squeeze = Conv(channels, int(channels // t), 1, act=nn.ReLU())
        self.excitation = Conv(int(channels // t), channels, 1, act=nn.Sigmoid())

    def forward(self, x):
        out = self.pool(x)
        out = self.squeeze(out)
        out = self.excitation(out)
        return x * out


class ShuffleBlock(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size, stride=1, act=nn.ReLU()):
        super(ShuffleBlock, self).__init__()
        self.act = act
        judge = (stride != 1) or (in_c != out_c)
        self.branch1 = self.make_branch_left(in_c, out_c // 2, kernel_size, stride) if judge else False
        self.branch2 = self.make_branch_right(in_c if judge else in_c // 2, out_c // 2, kernel_size, stride)

    def forward(self, x):
        if self.branch1:
            left, right = self.branch1(x), x
        else:
            left, right = paddle.split(x, 2, axis=1)
        right = self.branch2(right)
        x = paddle.concat([left, right], axis=1)
        return self.channel_shuffle(x)

    def make_branch_left(self, in_channels, out_channels, kernel_size=3, stride=2):
        return nn.Sequential(
            Conv(in_channels, in_channels, kernel_size, stride=stride, groups=in_channels, bn=True),
            Conv(in_channels, out_channels, 1, stride=1, groups=1, bn=True, act=self.act)
        )

    def make_branch_right(self, in_channels, out_channels, kernel_size=3, stride=2):
        return nn.Sequential(
            Conv(in_channels, out_channels, 1, stride=1, groups=1, bn=True, act=self.act),
            Conv(out_channels, out_channels, kernel_size, stride=stride, groups=out_channels, bn=True),
            Conv(out_channels, out_channels, 1, stride=1, groups=1, bn=True, act=self.act)
        )

    @staticmethod
    def channel_shuffle(x):
        batch_size, channels, height, width = x.shape
        x = paddle.reshape(x, (batch_size, 2, channels // 2, height, width))
        x = paddle.transpose(x, (0, 2, 1, 3, 4))
        x = paddle.reshape(x, (batch_size, channels, height, width))
        return x


class BottleBlock(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size, stride, scale=4, act=nn.ReLU()):
        super(BottleBlock, self).__init__()
        temp_c = int(out_c * scale)
        self.up_conv = Conv(in_c, temp_c, 1, 1, bn=True, act=act)
        self.dw_conv = Conv(temp_c, temp_c, kernel_size, stride, groups=temp_c, bn=True, act=act)
        self.down_conv = Conv(temp_c, out_c, 1, 1, bn=True)

        self.short_conv = Conv(in_c, out_c, 3, stride, bn=True) if in_c != out_c or stride != 1 else None
        self.act = act

    def forward(self, x):
        out = self.up_conv(x)
        out = self.dw_conv(out)
        out = self.down_conv(out)
        out = self.short_conv(x) + out if self.short_conv else x + out
        return self.act(out)


class DeBottleBlock(nn.Layer):
    def __init__(self, in_c, out_c, kernel_size, stride, scale=4, act=nn.ReLU()):
        super(DeBottleBlock, self).__init__()
        temp_c = int(out_c * scale)
        self.up_conv = Conv(in_c, temp_c, 1, 1, bn=True, act=act)
        self.dw_conv = DeConv(temp_c, temp_c, kernel_size, stride, group=temp_c, bn=True, act=act)
        self.down_conv = Conv(temp_c, out_c, 1, 1, bn=True)

        self.short_conv = DeConv(in_c, out_c, kernel_size, stride, bn=True) if in_c != out_c or stride != 1 else None
        self.act = act

    def forward(self, x):
        out = self.up_conv(x)
        out = self.dw_conv(out)
        out = self.down_conv(out)
        out = self.short_conv(x) + out if self.short_conv else x + out
        return self.act(out)


class MultiBlock(nn.Layer):
    def __init__(self, n, in_channels, out_channels, kernel_size, stride, twins=None, t=4, act=nn.Hardswish()):
        super(MultiBlock, self).__init__()
        self.layers = []
        self.twins = twins
        if not twins:
            for i in range(n):
                self.layers.append(ShuffleBlock(in_channels, out_channels, kernel_size, stride, act=act))
                in_channels = out_channels
                stride = 1
            self.layers.append(SEBlock(out_channels, t))
        else:
            for i in range(n//2):
                self.layers.append(ShuffleBlock(in_channels, twins, kernel_size, 1, act=act))
                in_channels = twins
            self.layers.append(SEBlock(in_channels, t))
            for i in range(n//2, n):
                self.layers.append(ShuffleBlock(in_channels, out_channels, kernel_size, stride, act=act))
                in_channels = out_channels
                stride = 1
            self.layers.append(SEBlock(out_channels, t))
        self.layers = nn.LayerList(self.layers)

    def forward(self, x):
        if self.twins:
            for i in range(len(self.layers) // 2):
                x = self.layers[i](x)
            out = x
            for i in range(len(self.layers) // 2, len(self.layers)):
                out = self.layers[i](out)
            return x, out
        else:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            return x


if __name__ == "__main__":
    model = MultiBlock(4, 16, 24, 3, 2, True)
    paddle.summary(model, (None, 16, 64, 64))
