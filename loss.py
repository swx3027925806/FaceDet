import paddle
import paddle.nn as nn


class CenterLoss(nn.Layer):
    def __init__(self, alpha=2, beta=4):
        super(CenterLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1loss = nn.L1Loss(reduction='sum')

    def forward(self, pre_heatmap, pre_h_and_w, pre_offset, tar_heatmap, tar_h_and_w, tar_offset):
        heatmap, N = self.focal_loss(pre_heatmap, tar_heatmap)
        offset = self.l1loss(pre_offset[tar_offset != 0], tar_offset[tar_offset != 0]) / N
        h_and_w = self.l1loss(pre_h_and_w[tar_offset != 0], tar_h_and_w[tar_offset != 0]) / N
        # print(heatmap, h_and_w, offset)
        return heatmap + h_and_w + offset

    def focal_loss(self, predict, target):
        predict = nn.functional.sigmoid(predict)
        N = paddle.sum(paddle.cast(target == 1.0, 'float32'))
        N = 1 if N == 0 else N
        predict = paddle.clip(predict, 1e-12, 1 - (1e-5))
        temp = paddle.log(1 - predict[target != 1.0]) / N
        trueloss = - paddle.pow((1 - predict[target == 1.0]), self.alpha) * paddle.log(predict[target == 1.0]) / N
        falseloss = - paddle.pow((1 - target[target != 1.0]), self.beta) * paddle.pow(predict[target != 1.0], self.alpha) * temp
        loss = paddle.sum(trueloss) + paddle.sum(falseloss)
        return loss, N
