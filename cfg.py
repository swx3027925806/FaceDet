import train

config = {
    "iters": 100000,
    "pretrain": "model/CenterNetShuffleNetHourglass/1_7_4_46000",
    "network": {
        "alpha": 1,
        "kernel_size": 7
    },
    "optim": {
        "lr": 0.001,
        "eta_min": 0.
    },
    "train": {
        "batch_size": 16,
        "info": 100,
        "eval_iters": 2000,
        "save_path": "save_model/iter",
        "list_path": "dataset/train_val.json",
        "transform": [
            {
                "type": "ReadImage",
                "root": "dataset"
            },
            {
                "type": "RandFlip",
                "p": 0.5
            },
            {
                "type": "RandHSV"
            },
            {
                "type": "RandomScale",
                "scale": [0.5, 2]
            },
            {
                "type": "RandomCrop",
                "crop_size": [512, 512]
            },
            {
                "type": "CenterNet"
            },
            {
                "type": "NormalizeImage"
            }
        ]
    },
    "val": {
        "batch_size": 16,
        "list_path": "dataset/val.json",
        "transform": [
            {
                "type": "ReadImage",
                "root": "dataset"
            },
            {
                "type": "RandomCrop",
                "crop_size": [512, 512]
            },
            {
                "type": "CenterNet"
            },
            {
                "type": "NormalizeImage"
            }
        ]
    },
    "assess": {}
}


if __name__ == "__main__":
    engine = train.Engine(config)
    engine.infer("test_image/big.jpg", "test_image/biganswer.jpg")
    engine.infer("test_image/side.jpg", "test_image/sideanswer.jpg")
    engine.infer("test_image/small.jpg", "test_image/smallanswer.jpg")
