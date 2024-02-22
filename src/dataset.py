import cv2
import random
import numpy as np
import torchvision


class TrainDataset:
    def __init__(self, root, unknown_label = 0, img_size = 32, download = False):
        self.dataset = torchvision.datasets.MNIST(
            root = root,
            train = True,
            download = download
        )
        self.unknown_label = unknown_label
        self.img_size = img_size

    def __len__(self):
        return len(self.dataset)
    
    def preprocess(self, img):
        img = np.array(img)
        img = cv2.resize(
            img,
            dsize = (self.img_size, self.img_size),
            interpolation = cv2.INTER_CUBIC
        )
        img = (img - 127.5) / 127.5
        img = img[None, :, :]
        return img
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        if label == self.unknown_label:
            idx = random.randrange(self.__len__())
            image, label = self.dataset[idx]

        return dict(
            image = self.preprocess(image),
            label = label
        )
    

class TestDataset(TrainDataset):
    def __init__(self, root, unknown_label = 0, img_size = 32, download = False):
        self.dataset = torchvision.datasets.MNIST(
            root = root,
            train = False,
            download = download
        )
        self.unknown_label = unknown_label
        self.img_size = img_size

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        return dict(
            image = self.preprocess(image),
            label = label
        )