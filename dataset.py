import os
import random
import tools
import cv2
import copy
import json
import paddle
import numpy as np
import paddle.io as io


class Datareader(io.Dataset):
    def __init__(self, parameters):
        super(Datareader, self).__init__()
        file = open(parameters["list_path"], 'r')
        self.data = json.load(file)
        self.trans = []

        # 将数据增强的操作进行配置
        for item in parameters["transform"]:
            self.trans.append(eval(item["type"])(item))

    def __getitem__(self, item):
        data = copy.deepcopy(self.data[item])
        for trans in self.trans:
            data = trans(data)
        return paddle.to_tensor(data["image"], dtype=paddle.float32), \
               paddle.to_tensor(data["heatmap"], dtype=paddle.float32), \
               paddle.to_tensor(data["h_and_w"], dtype=paddle.float32), \
               paddle.to_tensor(data["offset"], dtype=paddle.float32)

    def __len__(self):
        return len(self.data)


# 设定在参数传递时，以字典形式封装data[img], data[target]


class ReadImage:
    def __init__(self, parameters):
        self.root = parameters["root"]
        pass

    def __call__(self, data):
        data["image"] = cv2.imread(os.path.join(self.root, data["path"]))
        data["shape"] = data["image"].shape[:2]
        if "objs" in data.keys():
            data["target"] = data["objs"]
        return data


class RandFlip:
    def __init__(self, parameters):
        self.p = parameters["p"] if "p" in parameters.keys() else 0.5
        pass

    def __call__(self, data):
        if random.random() < self.p:
            data["image"] = data["image"][:, ::-1, :]
            for i in range(len(data["target"])):
                data["target"][i]['x'] = data["shape"][1] - data["target"][i]['x'] - data["target"][i]['w']
        return data


class RandHSV:
    def __init__(self, parameters):
        self.h = parameters["h"] if "h" in parameters.keys() else [-15, 15]     # 色调
        self.s = parameters["s"] if "s" in parameters.keys() else [0.5, 2.0]    # 饱和度
        self.v = parameters["v"] if "v" in parameters.keys() else [0.1, 1.0]    # 明度
        pass

    def __call__(self, data):
        image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2HSV)
        image[:, :, 0] = (image[:, :, 0] + random.randint(*self.h) + 360) % 360
        image[:, :, 1] = image[:, :, 1] * (random.random() * (self.s[1] - self.s[0]) + self.s[0])
        image[:, :, 2] = image[:, :, 2] * (random.random() * (self.v[1] - self.v[0]) + self.v[0])
        data["image"] = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return data


class StandardScale:
    def __init__(self, parameters):
        self.edge = parameters["long_edge"] if "long_edge" in parameters.keys() else 1024

    def __call__(self, data):
        pad_y = (data["shape"][1] + 1 - data["shape"][0]) // 2 if (data["shape"][1] + 1 - data["shape"][0]) // 2 > 0 else 0
        pad_x = (data["shape"][0] + 1 - data["shape"][1]) // 2 if (data["shape"][0] + 1 - data["shape"][1]) // 2 > 0 else 0
        data["image"] = cv2.copyMakeBorder(data["image"], pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT,
                                           value=(128, 128, 128))
        lens = data["image"].shape[0]
        data["image"] = cv2.resize(data["image"], (self.edge, self.edge), interpolation=cv2.INTER_CUBIC)
        data["shape"] = data["image"].shape
        scale = data["shape"][0] / lens
        if "target" in data.keys():
            for i in range(len(data["target"])):
                data["target"][i]['x'] = int((data["target"][i]['x'] + pad_x) * scale)
                data["target"][i]['y'] = int((data["target"][i]['y'] + pad_y) * scale)
                data["target"][i]['w'] = int(data["target"][i]['w'] * scale)
                data["target"][i]['h'] = int(data["target"][i]['h'] * scale)
        return data


class RandomScale:
    def __init__(self, parameters):
        self.scale = parameters["scale"] if "scale" in parameters.keys() else [0.2, 2.0]
        self.deform = parameters["deform"] if "deform" in parameters.keys() else [3/4, 4/3]

    def __call__(self, data):
        deform = np.sqrt(random.random() * (self.deform[1] - self.deform[0]) + self.deform[0])
        scale = np.sqrt(random.random() * (self.scale[1] - self.scale[0]) + self.scale[0])
        h, w = deform * scale, scale / deform
        data["image"] = cv2.resize(data["image"], (int(data["shape"][1] * w), int(data["shape"][0] * h)),
                                   interpolation=cv2.INTER_CUBIC)
        data["shape"] = data["image"].shape[:2]
        for i in range(len(data["target"])):
            data["target"][i]['x'] = int(data["target"][i]['x'] * w)
            data["target"][i]['y'] = int(data["target"][i]['y'] * h)
            data["target"][i]['w'] = int(data["target"][i]['w'] * w)
            data["target"][i]['h'] = int(data["target"][i]['h'] * h)
        return data


class RandomCrop:
    def __init__(self, parameters):
        self.crop_size = parameters["crop_size"] if "crop_size" in parameters.keys() else [512, 512]

    def __call__(self, data):
        pad_y = (self.crop_size[0] + 1 - data["shape"][0]) // 2 if (self.crop_size[0] + 1 - data["shape"][0]) // 2 > 0 else 0
        pad_x = (self.crop_size[1] + 1 - data["shape"][1]) // 2 if (self.crop_size[1] + 1 - data["shape"][1]) // 2 > 0 else 0
        data["image"] = cv2.copyMakeBorder(data["image"], pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(128,128,128))
        data["shape"] = data["image"].shape[:2]
        y = random.randint(0, data["shape"][0] - self.crop_size[0])
        x = random.randint(0, data["shape"][1] - self.crop_size[1])
        data["image"] = data["image"][y:y+self.crop_size[0], x:x+self.crop_size[1], :]
        data["shape"] = data["image"].shape
        target = []
        for i in range(len(data["target"])):
            iou = self.iou(data["target"][i], {"h": self.crop_size[0], "w": self.crop_size[1], "x": x, "y": y})
            if iou:
                data["target"][i]['x'] = data["target"][i]['x'] + pad_x - x
                data["target"][i]['y'] = data["target"][i]['y'] + pad_y - y
                temp = copy.deepcopy(data["target"][i])
                if data["target"][i]['x'] < 0:
                    data["target"][i]['w'] = data["target"][i]['w'] + data["target"][i]['x']
                    data["target"][i]['x'] = 0
                if data["target"][i]['y'] < 0:
                    data["target"][i]['h'] = data["target"][i]['h'] + data["target"][i]['y']
                    data["target"][i]['y'] = 0
                if data["target"][i]['x'] + data["target"][i]['w'] > self.crop_size[1]:
                    data["target"][i]['w'] = self.crop_size[1] - data["target"][i]['x']
                if data["target"][i]['y'] + data["target"][i]['h'] > self.crop_size[0]:
                    data["target"][i]['h'] = self.crop_size[0] - data["target"][i]['y']
                if self.iou(temp, data["target"][i]):
                    target.append(data["target"][i])
        data["target"] = target
        return data

    def iou(self, obj, crop):
        area1 = obj["w"] * obj["h"]
        area2 = crop["w"] * crop["h"]
        overlapWidth = min(obj["x"] + obj["w"], crop["x"] + crop["w"]) - max(obj["x"], crop["x"])
        overlapHeight = min(obj["y"] + obj["h"], crop["y"] + crop["h"]) - max(obj["y"], crop["y"])
        overlapArea = max(overlapWidth, 0) * max(overlapHeight, 0)
        return overlapArea / (area1 + area2 - overlapArea)


class Display:
    def __init__(self, parameters):
        self.save_path = parameters["save_path"]
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        pass

    def __call__(self, data):
        image = data["image"]
        predict = tools.decode_bbox(paddle.to_tensor(data["heatmap"][None, :], dtype=paddle.float32),
                                    paddle.to_tensor(data["h_and_w"][None, :], dtype=paddle.float32),
                                    paddle.to_tensor(data["offset"][None, :], dtype=paddle.float32))
        for item in predict[0]:
            image = cv2.rectangle(image, (int(item["x"] * 512), int(item["y"] * 512)), (int((item["x"] + item["w"]) * 512), int((item["y"] + item["h"]) * 512)), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(self.save_path, data["path"].split('/')[-1].split('.')[0]) + ".png", image)
        return data


class CenterNet:
    def __init__(self, parameters):
        self.min_overlap = parameters["min_overlap"] if "min_overlap" in parameters.keys() else 0.7

    def __call__(self, data):
        data["heatmap"], data["h_and_w"], data["offset"] = self.obj2heatmap(data["target"], data["shape"], 4)
        return data

    def obj2heatmap(self, target, image_size, scale):
        out_size = (image_size[0] // scale, image_size[1] // scale)
        heatmap = np.zeros((out_size[0], out_size[1]))
        h_and_w = np.zeros((2, out_size[0], out_size[1]))
        offset = np.zeros((2, out_size[0], out_size[1]))
        for item in target:
            r = self.calculate_r(item)
            center_point = (int((item['y'] + item['h'] // 2) // scale), int((item['x'] + item['w'] // 2) // scale))
            if center_point[0] >= out_size[0] or center_point[1] >= out_size[1]:
                continue
            h_and_w[0, center_point[0], center_point[1]] = item['h'] / image_size[0]
            h_and_w[1, center_point[0], center_point[1]] = item['w'] / image_size[1]
            offset[0, center_point[0], center_point[1]] = (item['y'] + item['h'] / 2) / scale - center_point[0]
            offset[1, center_point[0], center_point[1]] = (item['x'] + item['w'] / 2) / scale - center_point[1]
            heatmap = self.draw_heatmap(heatmap, center_point, int(r))
        return heatmap[None, :, :], h_and_w, offset

    def calculate_r(self, obj):
        """
        计算可偏移角度
        :param obj: 对象的属性
        :return: 高斯圆半径
        """
        min_overlap = self.min_overlap
        height, width = obj["h"], obj["w"]
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        # 情况二
        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        # 情况一
        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        # np.finfo(h.dtype).eps eps为一个接近0的小数，方式除数为0出现Nan
        return h

    def draw_heatmap(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        y, x = int(center[0]), int(center[1])
        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


class CLAHE:
    def __init__(self, parameters):
        self.clipLimit = parameters["clipLimit"] if "clipLimit" in parameters.keys() else (2.0, 5.0)
        self.tileGridSize = parameters["tileGridSize"] if "tileGridSize" in parameters.keys() else (4, 16)
        pass

    def __call__(self, data):
        image = cv2.cvtColor(data["image"], cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(image)

        clipLimit = random.random() * (self.clipLimit[1] - self.clipLimit[0]) + self.clipLimit[0]
        tileGridSize = random.randint(*self.tileGridSize)

        clahe = cv2.createCLAHE(clipLimit, (tileGridSize, tileGridSize))

        l = clahe.apply(l)

        image = cv2.merge([l, a, b])
        data["image"] = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return data


class NormalizeImage:
    def __init__(self, parameters):
        self.scale = parameters["scale"] if "scale" in parameters.keys() else 1 / 255
        self.mean = np.array(parameters["mean"] if "mean" in parameters.keys() else [0.485, 0.456, 0.406])
        self.std = np.array(parameters["std"] if "std" in parameters.keys() else [0.229, 0.224, 0.225])

    def __call__(self, data):
        data["image"] = (data["image"].astype('float32') * self.scale - self.mean) / self.std
        data["image"] = data["image"].transpose((2, 0, 1))
        return data


if __name__ == "__main__":
    config = {
        "list_path": "data/val.json",
        "transform": [
            {
                "type": "ReadImage",
                "root": "data"
            },
            {
                "type": "StandardScale",
                "long_edge": 512
            },
            {
                "type": "CenterNet"
            },
            {
                "type": "Display",
                "save_path": "test_image"
            }
        ]
    }
    dataset = io.DataLoader(Datareader(config), batch_size=1)
    for batch_id, data in enumerate(dataset()):
        # print("image", data[0].shape, data[0].max().numpy(), data[0].min().numpy())
        # print("heatmap", data[1].shape, data[1].max().numpy(), data[1].min().numpy())
        # print("h_and_w", data[2].shape, data[2].max().numpy(), data[2].min().numpy())
        # print("offset", data[3].shape, data[3].max().numpy(), data[3].min().numpy())
        pass
