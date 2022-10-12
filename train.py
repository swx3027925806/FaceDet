import time
from loss import CenterLoss
from dataset import *
import paddle.optimizer as optimizer
import paddle.optimizer.lr as lr
from Network.CenterNet import *


class Engine:
    def __init__(self, parameters):
        self.model = CenterNetShuffleNetHourglass(parameters["network"])
        self.lr = lr.CosineAnnealingDecay(parameters["optim"]["lr"], T_max=parameters["iters"], eta_min=parameters["optim"]["eta_min"])
        self.sum_iters = parameters["iters"]
        weight_decay = paddle.regularizer. L2Decay(0.0004)
        self.optim = optimizer.Adam(self.lr, parameters=self.model.parameters(), weight_decay=weight_decay)
        self.assess_cfg = parameters["assess"]
        self.model, self.optim = self.load_model(parameters["pretrain"], self.model, self.optim)

        self.train_cfg = parameters["train"]
        self.val_cfg = parameters["val"]

    def train(self):
        self.model.train()
        iters = 0
        best = 0
        loss_set = []
        loss_func = CenterLoss()
        dataset = Datareader(self.train_cfg)
        dataloader = io.DataLoader(dataset, batch_size=self.train_cfg["batch_size"], shuffle=True)
        time_a = time.time()
        while True:
            for batch_id, data in enumerate(dataloader()):
                iters += 1
                self.optim.clear_grad()
                heatmap, h_and_w, offset = self.model(data[0])
                loss_value = loss_func(heatmap, h_and_w, offset, data[1], data[2], data[3])
                loss_value.backward()
                self.optim.step()
                loss_set.append(loss_value.numpy())
                if iters % self.train_cfg["info"] == 0:
                    time_b = time.time()
                    self.display(time_b - time_a, self.lr.get_lr(), np.array(loss_set).mean(), iters, self.sum_iters, self.train_cfg["info"])
                    time_a = time_b
                    loss_set.clear()
                self.lr.step()
                if iters % self.train_cfg["eval_iters"] == 0:
                    self.save_model(self.train_cfg["save_path"], self.model, self.optim, iters)
                    loss, map_ = self.eval()
                    if best < map_:
                        self.save_model(self.train_cfg["save_path"], self.model, self.optim, iters, best=True)
                    best = map_
                    print("loss:%7.5f now_map:%7.5f best_map:%7.5f" % (loss, map_, best))
                if iters > self.sum_iters:
                    break

    @paddle.no_grad()
    def eval(self):
        self.model.eval()
        ap = tools.Average(self.assess_cfg)
        loss_set = []
        loss_func = CenterLoss()
        dataset = Datareader(self.val_cfg)
        dataloader = io.DataLoader(dataset, batch_size=self.val_cfg["batch_size"], shuffle=False)
        for batch_id, data in enumerate(dataloader()):
            heatmap, h_and_w, offset = self.model(data[0])
            loss_value = loss_func(heatmap, h_and_w, offset, data[1], data[2], data[3])
            loss_set.append(loss_value.numpy())
            ap.collect(heatmap, h_and_w, offset, data[1], data[2], data[3])

        self.model.train()
        return np.array(loss_set).mean(), ap.calculate()

    def infer(self, image_path, save_path):
        read_image = ReadImage({"root": ""})
        standardscale = StandardScale({"long_edge": 1024})
        normalize = NormalizeImage({})
        image = standardscale(read_image({"path": image_path}))
        data = normalize(copy.deepcopy(image))
        heatmap, h_and_w, offset = self.model(paddle.to_tensor(data["image"][None, :, :, :], dtype=paddle.float32))
        predict = tools.decode_bbox(heatmap, h_and_w, offset, confidence=0.01)
        image = image["image"]
        for item in predict[0]:
            # print(int((item["x"] + item["w"]) * 1024), int((item["y"] + item["h"]) * 1024))
            image = cv2.rectangle(image, (int((item["x"] + item["w"]*0.1) * 1024), int((item["y"] + item["h"]*0.1) * 1024)),
                                  (int((item["x"] + item["w"]*0.9) * 1024), int((item["y"] + item["h"]*0.9) * 1024)), (0, 0, 255), 1)
        print(save_path)
        cv2.imwrite(save_path, image)

    @staticmethod
    def load_model(path, model, optim):
        if path:
            model.set_state_dict(paddle.load(path + ".pdparams"))
            # optim.set_state_dict(paddle.load(path + '.pdopt'))
        return model, optim

    @staticmethod
    def save_model(path, model, optim, iters, best=False):
        path = path + 'Best' if best else path
        paddle.save(model.state_dict(), path + '_%d.pdparams' % iters)
        paddle.save(optim.state_dict(), path + '_%d.pdopt' % iters)

    def nms(self, predict, size):
        heat_max = paddle.nn.functional.max_pool2d(predict, kernel_size=size, stride=1, padding=size // 2)
        heat_mask = paddle.cast((heat_max == predict), 'float32')
        return heat_mask * predict

    @staticmethod
    def display(times, lr_value, loss_value, iters, sum_iter, interval):
        process = iters / sum_iter
        times = times * (sum_iter - iters) / interval
        h = times // 3600
        m = (times % 3600) // 60
        print("Process:%7.5f Iters:%6d/%6d RTime:%3dH%3dMin lr:%7.5f loss:%7.5f" % (
            process, iters, sum_iter, h, m, lr_value, loss_value))