import paddle

from Network.BackBone import *


class Heatmap(nn.Layer):
    def __init__(self, in_c, act=nn.ReLU()):
        super(Heatmap, self).__init__()
        self.heatmap_conv1 = Conv(in_c, in_c, kernel_size=3, stride=1, act=act)
        self.heatmap_conv2 = Conv(in_c, 1, kernel_size=1, stride=1)

        self.wh_conv1 = Conv(in_c, in_c, kernel_size=3, stride=1, act=act)
        self.wh_conv2 = Conv(in_c, 2, kernel_size=1, stride=1)

        self.offset_conv1 = Conv(in_c, in_c, kernel_size=3, stride=1, act=act)
        self.offset_conv2 = Conv(in_c, 2, kernel_size=1, stride=1)

    def forward(self, x):
        heatmap = self.heatmap_conv2(self.heatmap_conv1(x))
        wh = self.wh_conv2(self.wh_conv1(x))
        offset = self.offset_conv2(self.offset_conv1(x))
        return heatmap, wh, offset


class CenterNetShuffleNetV2(nn.Layer):
    def __init__(self, parameters, act=nn.ReLU()):
        super(CenterNetShuffleNetV2, self).__init__()
        self.backbone = ShuffleNetV2(parameters)
        self.conv1 = Conv(self.backbone.end_channels, 256, 3, 1, bn=True, act=act)
        self.deco1 = DeConv(256, 256, 4, 2, 1, bn=True, act=act)

        self.conv2 = Conv(256, 128, 3, 1, bn=True, act=act)
        self.deco2 = DeConv(128, 128, 4, 2, 1, bn=True, act=act)

        self.conv3 = Conv(128, 64, 3, 1, bn=True, act=act)
        self.deco3 = DeConv(64, 64, 4, 2, 1, bn=True, act=act)

        self.heatmap = Heatmap(64)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deco1(self.conv1(x))
        x = self.deco2(self.conv2(x))
        x = self.deco3(self.conv3(x))
        return self.heatmap(x)


class CenterNetShuffleNetV2Plus(nn.Layer):
    def __init__(self, parameters, act=nn.ReLU()):
        super(CenterNetShuffleNetV2Plus, self).__init__()
        self.backbone = ShuffleNetV2Plus(parameters)
        self.t = parameters["t"] if "t" in parameters else 6
        self.conv1 = ShuffleBlock(self.backbone.cfg * 8, self.backbone.cfg * 8, 7, 1)
        self.deco1 = DeConv(self.backbone.cfg * 8, self.backbone.cfg * 4, 4, 2, 1, bn=True, act=act)
        self.se1 = SEBlock(self.backbone.cfg * 4, self.t)

        self.conv2 = ShuffleBlock(self.backbone.cfg * 8, self.backbone.cfg * 8, 7, 1)
        self.deco2 = DeConv(self.backbone.cfg * 8, self.backbone.cfg * 2, 4, 2, 1, bn=True, act=act)
        self.se2 = SEBlock(self.backbone.cfg * 2, self.t)

        self.conv3 = ShuffleBlock(self.backbone.cfg * 4, self.backbone.cfg * 4, 7, 1)
        self.deco3 = DeConv(self.backbone.cfg * 4, self.backbone.cfg, 4, 2, 1, bn=True, act=act)
        self.se3 = SEBlock(self.backbone.cfg, self.t)

        self.heatmap = Heatmap(self.backbone.cfg)

    def forward(self, x):
        out1, out2, out3 = self.backbone(x)
        out3 = self.se1(self.deco1(self.conv1(out3)))
        out2 = self.se2(self.deco2(self.conv2(paddle.concat((out2, out3), axis=1)))) # [batch, C, H, W]
        out1 = self.se3(self.deco3(self.conv3(paddle.concat((out1, out2), axis=1))))
        return self.heatmap(out1)


class CenterNetShuffleNetHourglass(nn.Layer):
    def __init__(self, parameters, act=nn.Hardswish()):
        super(CenterNetShuffleNetHourglass, self).__init__()
        self.backbone = ShuffleNetHourglass(parameters)
        self.kernel_size = parameters["kernel_size"] if "kernel_size" in parameters else 7
        self.t = parameters["t"] if "t" in parameters else 4
        self.stage1_merge1 = BottleBlock(self.backbone.channels * 32, self.backbone.channels * 8, self.kernel_size, 1, act=act)
        self.stage1_deconv = DeBottleBlock(self.backbone.channels * 8, self.backbone.channels * 8, 4, 2, act=act)


        self.stage2_merge1 = BottleBlock(self.backbone.channels * 32, self.backbone.channels * 8, self.kernel_size, 1, act=act)
        self.stage2_merge2 = BottleBlock(self.backbone.channels * 16, self.backbone.channels * 8, self.kernel_size, 1, act=act)
        self.stage2_deconv = DeBottleBlock(self.backbone.channels * 8, self.backbone.channels * 4, 4, 2, act=act)

        self.stage3_merge1 = BottleBlock(self.backbone.channels * 16, self.backbone.channels * 4, self.kernel_size, 1, act=act)
        self.stage3_merge2 = BottleBlock(self.backbone.channels * 8, self.backbone.channels * 4, self.kernel_size, 1, act=act)
        self.stage3_deconv = DeBottleBlock(self.backbone.channels * 4, self.backbone.channels * 2, 4, 2, act=act)

        self.stage4_merge1 = BottleBlock(self.backbone.channels * 8, self.backbone.channels * 2, self.kernel_size, 1, act=act)
        self.stage4_merge2 = BottleBlock(self.backbone.channels * 4, self.backbone.channels * 2, self.kernel_size, 1, act=act)

        self.heatmap = Heatmap(self.backbone.channels * 2)

    def forward(self, x):
        out1, out2, out3, out4 = self.backbone(x)
        out = self.stage1_deconv(self.stage1_merge1(out4))
        out = self.stage2_deconv(self.stage2_merge2(paddle.concat((self.stage2_merge1(out3), out), axis=1)))
        out = self.stage3_deconv(self.stage3_merge2(paddle.concat((self.stage3_merge1(out2), out), axis=1)))
        out = self.stage4_merge2(paddle.concat((self.stage4_merge1(out1), out), axis=1))
        return self.heatmap(out)


if __name__ == "__main__":
    # model = CenterNetShuffleNetV2({"alpha": 1})
    # paddle.summary(model, (None, 3, 512, 512))
    # model = CenterNetShuffleNetV2Plus({"alpha": 1})
    # paddle.summary(model, (None, 3, 512, 512))
    model = CenterNetShuffleNetHourglass({"alpha": 1})
    paddle.summary(model, (None, 3, 512, 512))
