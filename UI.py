import copy
import os
import sys
import cv2
import time
import paddle
from dataset import *
from Network.CenterNet import CenterNetShuffleNetV2, CenterNetShuffleNetV2Plus, CenterNetShuffleNetHourglass
from UIWindows import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets


class MainForm(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)

        self.network_dir = "model"
        self.network_list_path = os.listdir(self.network_dir)
        self.network_path = None
        self.model_path = None
        self.path = None

        self.path_buttom.clicked.connect(self.choose_image_dir)
        self.network_buttom.clicked.connect(self.choose_network)
        self.model_buttom.clicked.connect(self.choose_model)
        self.network_list.addItems(self.network_list_path)
        self.path_list.clicked.connect(self.detection)

        self.model = None
        self.qlist = []

    def choose_image_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./")
        self.path_text.setText(directory)
        self.path = directory
        self.qlist = os.listdir(directory)

        slm = QtCore.QStringListModel()
        slm.setStringList(self.qlist)
        self.path_list.setModel(slm)

    def choose_network(self):
        self.model_list.clear()
        self.network_path = self.network_list.currentText()
        self.model_list.addItems(os.listdir(os.path.join(self.network_dir, self.network_path)))

    def choose_model(self):
        self.model_path = self.model_list.currentText()
        if self.network_path == "Harr":
            self.model = Harr(os.path.join(self.network_dir, self.network_path, self.model_path))
        else:
            self.model = Network(self.network_dir, self.network_path, self.model_path)

    def detection(self, item):
        path = os.path.join(self.path, self.qlist[item.row()])
        self.model.predict(path)
        image = self.model.image_info(self.retangle_check.isChecked())
        pad_l = (1280 - image.shape[1]) // 2 if image.shape[1] < 1280 else 0
        pad_t = (720 - image.shape[0]) // 2 if image.shape[0] < 720 else 0
        image = cv2.copyMakeBorder(image, pad_t, pad_t, pad_l, pad_l, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        cv2.imwrite('resource/predict.png', image)
        self.dislpay_image.setPixmap(QtGui.QPixmap("resource/predict.png"))
        self.dislpay_image.show()
        self.base_info.setText(self.model.text_info())


class Harr:
    def __init__(self, model_path):
        self.raw_image = None
        self.new_image = None
        self.face_info = []
        self.harr = cv2.CascadeClassifier(model_path)
        pass

    def predict(self, path):
        a = time.time()
        self.raw_image = cv2.imread(path)
        stand = StandardScale({"long_edge": 1280})
        self.raw_image = stand({"image": self.raw_image, "shape": self.raw_image.shape[:2]})["image"]
        gray = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        face = self.harr.detectMultiScale(gray, 1.3, 5)
        self.face_info = ["time:%5.3fs" % (time.time() - a)]
        self.new_image = copy.deepcopy(self.raw_image)
        for (x, y, w, h) in face:
            cv2.rectangle(self.new_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.face_info.append("x:%3d y:%3d w:%3d h:%3d" % (x, y, w, h))

    def text_info(self):
        return "\n".join(self.face_info)

    def image_info(self, judge):
        return self.new_image if judge else self.raw_image


class Network:
    def __init__(self, root, model, model_path):
        self.raw_image = None
        self.new_image = None
        self.face_info = []
        cfg_split = model_path.split("_")
        cfg = {"alpha": int(cfg_split[0]), "kernel_size": int(cfg_split[1])}
        if len(cfg_split) > 4:
            cfg["t"] = int(cfg_split[2])
        self.model = eval(model)(cfg)
        self.model.set_state_dict(paddle.load(os.path.join(root, model, model_path)))

    def predict(self, path):
        a = time.time()
        read_image = ReadImage({"root": ""})
        standardscale = StandardScale({"long_edge": 1024})
        normalize = NormalizeImage({})
        image = standardscale(read_image({"path": path}))
        data = normalize(copy.deepcopy(image))
        heatmap, h_and_w, offset = self.model(paddle.to_tensor(data["image"][None, :, :, :], dtype=paddle.float32))
        predict = tools.decode_bbox(heatmap, h_and_w, offset, confidence=0.05)
        self.raw_image = image["image"]
        self.new_image = copy.deepcopy(self.raw_image)
        self.face_info = ["time:%5.3fs" % (time.time() - a)]
        for item in predict[0]:
            self.face_info.append("x:%3d y:%3d w:%3d h:%3d" % (int(item["x"] * 1024), int(item["y"] * 1024),
                                                               int(item["w"] * 1024), int(item["h"] * 1024)))
            self.new_image = cv2.rectangle(self.new_image, (int(item["x"] * 1024), int(item["y"] * 1024)),
                                          (int((item["x"] + item["w"]) * 1024), int((item["y"] + item["h"]) * 1024)), (0, 0, 255), 1)

    def text_info(self):
        return "\n".join(self.face_info)

    def image_info(self, judge):
        return self.new_image if judge else self.raw_image


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
