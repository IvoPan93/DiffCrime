import os

import numpy
from torch.utils.data import Dataset
from PIL import Image


class RiskMap(Dataset):
    def __init__(self, **kwargs):
        super().__init__()

        self.filenames = self.getFileNames()
        self.labels = dict()
        self.labels["file_names"] = self.filenames
        self._length = len(self.filenames)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        RISKMAP_PATH_171819 = "./data/171819RM/"
        RISKMAP_PATH_202122 = "./data/202122RM/"
        CON_IMG_MAP_256 = "./data/ConditionImageMap256/"
        CON_IMG_SATELLITE_256 = "./data/ConditionImageSatellite256/"

        example = dict()
        example["file_name"] = self.labels["file_names"][i]

        fn = example["file_name"] + ".npy"
        RM171819 = numpy.load(RISKMAP_PATH_171819 + fn)
        RM171819 = numpy.expand_dims(RM171819, axis=0).astype(numpy.float32)
        example["riskmap_171819"] = RM171819

        fn = example["file_name"] + ".npy"
        RM202122 = numpy.load(RISKMAP_PATH_202122 + fn)
        RM202122 = numpy.expand_dims(RM202122, axis=0).astype(numpy.float32)
        example["image"] = RM202122

        example["cond_map"] = self.preprocess_image(CON_IMG_MAP_256 + example["file_name"] + ".png")
        example["cond_satellite"] = self.preprocess_image(CON_IMG_SATELLITE_256 + example["file_name"] + ".png")
        return example

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = numpy.array(image).astype(numpy.uint8)
        image = (image / 127.5 - 1.0).astype(numpy.float32)  # 将像素都放缩到0-1之间
        return image

    def getFileNames(self):
        return []


class RiskMapTrain(RiskMap):
    def getFileNames(self):
        with open("./data/train_files.txt", "r") as f:
            relpaths = f.read().splitlines()
        return relpaths


class RiskMapValidation(RiskMap):
    def getFileNames(self):
        with open("./data/val_files.txt", "r") as f:
            relpaths = f.read().splitlines()
        return relpaths

class RiskMapTest(RiskMap):
    def getFileNames(self):
        with open("./data/test_files.txt", "r") as f:
            relpaths = f.read().splitlines()
        return relpaths