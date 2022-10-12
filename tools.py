import copy
import os
import json
import paddle
import numpy as np
import paddle.nn as nn


class MakeJson:
    def __init__(self, root_path, image_path, txt_path) -> None:
        self.root_path = root_path
        self.image_path = image_path
        self.txt_path = txt_path
        self.data = []

    def load_data(self):
        data_file = open(os.path.join(self.root_path, self.txt_path))
        while True:
            path = data_file.readline()
            if path == "":
                break
            photo = {"path": os.path.join(self.image_path, path.replace("\n", "")).replace("\\", "/"),
                     "count": int(data_file.readline().replace("\n", ""))}
            if photo["count"] == 0:
                data_file.readline()
                continue
            photo["objs"] = []
            for _ in range(photo["count"]):
                one = data_file.readline().replace("\n", "").split(" ")
                obj = {}
                obj["x"] = int(one[0])
                obj["y"] = int(one[1])
                obj["w"] = int(one[2])
                obj["h"] = int(one[3])
                photo["objs"].append(obj)
            self.data.append(photo)
        data_file.close()

    def save_json(self, path):
        with open('%s.json' % path, 'w', encoding='utf8') as json_file:
            json.dump(self.data, json_file)


@paddle.no_grad()
def nms_point(predict, kernel=5):
    """
    predict: 预测值为热力图的形式
    """
    heat_max = nn.functional.max_pool2d(predict, kernel_size=kernel, stride=1, padding=kernel // 2)
    heat_mask = paddle.cast((heat_max == predict), 'float32')
    return heat_mask * predict


def decode_bbox(heatmap_, w_and_h_, offset_, kernel=5, confidence=0.5):
    # 首先把标签转换成3D
    predict = []
    for item in range(heatmap_.shape[0]):
        heatmap = nms_point(heatmap_, kernel=kernel)[item].numpy()
        w_and_h = w_and_h_[item].numpy()
        offset = offset_[item].numpy()
        objs = []

        # 首先需要获得我们所需要的热力图标签
        obj_coord = np.argwhere(heatmap >= confidence)
        for obj in obj_coord:  # [0, y, x]
            score = heatmap[0, obj[1], obj[2]]
            center_x = (obj[2] + offset[1, obj[1], obj[2]]) / heatmap.shape[1]
            center_y = (obj[1] + offset[0, obj[1], obj[2]]) / heatmap.shape[2]
            w = w_and_h[1, obj[1], obj[2]]
            h = w_and_h[0, obj[1], obj[2]]
            print(center_x, center_y, w, h)
            objs.append({"score": score, "x": center_x - w / 2, "y": center_y - h / 2, "h": h, "w": w})
        predict.append(objs)
    return predict


class Average:
    def __init__(self, parameters):
        self.iou = parameters["iou"] if "iou" in parameters.keys() else (0.5,)
        self.confidence = parameters["confidence"] if "confidence" in parameters.keys() else 0.05
        self.kernel = parameters["kernel"] if "kernel" in parameters.keys() else 5
        self.static = []
        self.sum_obj = 0

    def collect(self, pre_heatmap, pre_w_and_h, per_offset, tar_heatmap, tar_w_and_h, tar_offset):
        predict = decode_bbox(nn.functional.sigmoid(pre_heatmap), pre_w_and_h, per_offset, self.kernel, self.confidence)
        target = decode_bbox(tar_heatmap, tar_w_and_h, tar_offset, self.kernel, self.confidence)
        target_ = copy.deepcopy(target)
        predict_ = copy.deepcopy(predict)
        for batch in range(len(target_)):
            self.sum_obj += len(target[batch])
            for pre in predict_[batch]:
                best = 0
                best_pre = None
                best_tar = None
                for tar in target_[batch]:
                    iou = self._iou(tar, pre)
                    if iou > best:
                        best = iou
                        best_pre = pre
                        best_tar = tar
                if best > 0:
                    self.static.append([best_pre["score"], best])
                    target_[batch].remove(best_tar)
                    predict_[batch].remove(best_pre)
                else:
                    continue
            for pre in predict_[batch]:
                self.static.append([pre["score"], 0])

    def calculate(self):
        if len(self.static) == 0:
            return 0
        else:
            static = np.array(self.static)
            static = static[static[:, 0].argsort()[::-1]]
            ap = np.zeros(len(self.iou))
            for threshold in range(len(self.iou)):
                temp = np.cumsum((static[:, 1] > threshold).astype(np.int64))
                p = np.nan_to_num(temp / np.arange(1, static.shape[0] + 1))
                r = np.nan_to_num(temp / self.sum_obj)

                for item in range(len(r)):
                    if item == 0:
                        ap[threshold] += r[item] * p[item]
                    elif r[item] != r[item - 1]:
                        ap[threshold] += (r[item] - r[item - 1]) * p[item]
            return ap.mean()

    def _iou(self, obj, crop):
        area1 = obj["w"] * obj["h"]
        area2 = crop["w"] * crop["h"]
        overlapWidth = min(obj["x"] + obj["w"], crop["x"] + crop["w"]) - max(obj["x"], crop["x"])
        overlapHeight = min(obj["y"] + obj["h"], crop["y"] + crop["h"]) - max(obj["y"], crop["y"])
        overlapArea = max(overlapWidth, 0) * max(overlapHeight, 0)
        return overlapArea / (area1 + area2 - overlapArea)


if __name__ == "__main__":
    train = MakeJson("data", "WIDER_train\images", "wider_face_train_bbx_gt.txt")
    train.load_data()
    train.save_json("train")

    val = MakeJson("data", "WIDER_val\images", "wider_face_val_bbx_gt.txt")
    val.load_data()
    val.save_json("val")