from Network.block import *


class ShuffleNetV2(nn.Layer):
    def __init__(self, parameters, act=nn.ReLU()):
        super(ShuffleNetV2, self).__init__()
        self.alpha = parameters["alpha"] if "alpha" in parameters else 1
        self.kernel_size = parameters["kernel_size"] if "kernel_size" in parameters else 3
        self.cfg = {0.5: 48, 1: 116, 1.5: 176, 2: 244}
        self.act = act
        self.conv = Conv(3, 24, self.kernel_size, 2, bn=True, act=self.act)
        self.pool = nn.MaxPool2D(2)
        self.block2 = self.make_layer(4, 24, self.cfg[self.alpha], self.kernel_size, 2)
        self.block3 = self.make_layer(8, self.cfg[self.alpha], 2 * self.cfg[self.alpha], self.kernel_size, 2)
        self.block4 = self.make_layer(4, 2 * self.cfg[self.alpha], 4 * self.cfg[self.alpha], self.kernel_size, 2)
        self.end_channels = 1024 if self.alpha < 2 else 2048
        self.conv2 = Conv(4 * self.cfg[self.alpha], self.end_channels, 1, bn=True, act=self.act)
        pass

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv2(x)
        return x

    @staticmethod
    def make_layer(n, in_channels, out_channels, kernel_size=3, stride=2):
        layers = []
        for i in range(n):
            layers.append(ShuffleBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
            stride = 1
        return nn.Sequential(*layers)


class ShuffleNetV2Plus(nn.Layer):
    def __init__(self, parameters, act=nn.ReLU()):
        super(ShuffleNetV2Plus, self).__init__()
        self.alpha = parameters["alpha"] if "alpha" in parameters else 1
        self.kernel_size = parameters["kernel_size"] if "kernel_size" in parameters else 7
        self.t = parameters["t"] if "t" in parameters else 6
        self.cfg = int(self.alpha * 64)
        self.conv = Conv(3, 32, self.kernel_size, 2, bn=True, act=act)
        self.block2 = self.make_layer(4, 32, self.cfg, self.kernel_size, 2)
        self.block3 = self.make_layer(4, self.cfg, self.cfg * 2, self.kernel_size, 2)
        self.block4 = self.make_layer(8, self.cfg * 2, self.cfg * 4, self.kernel_size, 2)
        self.block5 = self.make_layer(4, self.cfg * 4, self.cfg * 8, self.kernel_size, 2)
        pass

    def forward(self, x):
        """特征金字塔"""
        x = self.conv(x)
        x = self.block2(x)
        out1 = self.block3(x)
        out2 = self.block4(out1)
        out3 = self.block5(out2)
        return out1, out2, out3

    def make_layer(self, n, in_channels, out_channels, kernel_size=3, stride=2):
        layers = []
        for i in range(n):
            layers.append(ShuffleBlock(in_channels, out_channels, kernel_size, stride))
            in_channels = out_channels
            stride = 1
        layers.append(SEBlock(out_channels, self.t))
        return nn.Sequential(*layers)


class ShuffleNetHourglass(nn.Layer):
    def __init__(self, parameters, act=nn.Hardswish()):
        super(ShuffleNetHourglass, self).__init__()
        self.alpha = parameters["alpha"] if "alpha" in parameters else 1
        self.kernel_size = parameters["kernel_size"] if "kernel_size" in parameters else 7
        self.t = parameters["t"] if "t" in parameters else 4
        channels = int(16 * self.alpha)
        self.channels = channels
        self.conv = Conv(3, channels, 7, 1, 1, bn=True, act=act)
        self.stage1_1 = MultiBlock(2, channels, 2 * channels, self.kernel_size, stride=2, twins=channels, t=self.t, act=act)
        self.stage1_2 = MultiBlock(2, channels, 2 * channels, self.kernel_size, stride=2, twins=channels, t=self.t, act=act)
        self.stage1_3 = MultiBlock(2, channels, 2 * channels, self.kernel_size, stride=2, twins=channels, t=self.t, act=act)
        self.stage1_4 = MultiBlock(2, channels, 2 * channels, self.kernel_size, stride=2, t=self.t, act=act)

        self.stage2_1 = MultiBlock(2, 2 * channels, 4 * channels, self.kernel_size, stride=2, twins=2 * channels, t=self.t, act=act)
        self.stage2_2 = MultiBlock(2, 4 * channels, 4 * channels, self.kernel_size, stride=2, twins=2 * channels, t=self.t, act=act)
        self.stage2_3 = MultiBlock(2, 4 * channels, 4 * channels, self.kernel_size, stride=2, twins=2 * channels, t=self.t, act=act)
        self.stage2_4 = MultiBlock(2, 4 * channels, 4 * channels, self.kernel_size, stride=2, t=self.t, act=act)

        self.stage3_1 = MultiBlock(2, 4 * channels, 8 * channels, self.kernel_size, stride=2, twins=4 * channels, t=self.t, act=act)
        self.stage3_2 = MultiBlock(2, 8 * channels, 8 * channels, self.kernel_size, stride=2, twins=4 * channels, t=self.t, act=act)
        self.stage3_3 = MultiBlock(2, 8 * channels, 8 * channels, self.kernel_size, stride=2, twins=4 * channels, t=self.t, act=act)

        self.stage4_1 = MultiBlock(2, 8 * channels, 16 * channels, self.kernel_size, stride=2, twins=8 * channels, t=self.t, act=act)
        self.stage4_2 = MultiBlock(2, 16 * channels, 16 * channels, self.kernel_size, stride=2, twins=8 * channels, t=self.t, act=act)

        self.stage5_1 = MultiBlock(4, 16 * channels, 32 * channels, self.kernel_size, stride=2, twins=16 * channels, t=self.t, act=act)

    def forward(self, x):
        x = self.conv(x)
        out1, out_1 = self.stage1_1(x)
        out1, out_2 = self.stage1_2(out1)
        out1, out_3 = self.stage1_3(out1)
        out_4 = self.stage1_4(out1)

        out2, out_1 = self.stage2_1(out_1)
        out2, out_2 = self.stage2_2(paddle.concat((out2, out_2), axis=1))
        out2, out_3 = self.stage2_3(paddle.concat((out2, out_3), axis=1))
        out_4 = self.stage2_4(paddle.concat((out2, out_4), axis=1))

        out3, out_1 = self.stage3_1(out_1)
        out3, out_2 = self.stage3_2(paddle.concat((out3, out_2), axis=1))
        out3, out_3 = self.stage3_3(paddle.concat((out3, out_3), axis=1))
        out3 = paddle.concat((out3, out_4), axis=1)

        out4, out_1 = self.stage4_1(out_1)
        out4, out_2 = self.stage4_2(paddle.concat((out4, out_2), axis=1))
        out4 = paddle.concat((out4, out_3), axis=1)

        out5, out6 = self.stage5_1(out_1)
        out5 = paddle.concat((out5, out_2), axis=1)
        return out3, out4, out5, out6


if __name__ == "__main__":
    model = ShuffleNetHourglass({})
    paddle.summary(model, (None, 3, 224, 224))
