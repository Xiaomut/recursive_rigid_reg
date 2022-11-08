import os
import re
import sys
import torch
import numpy as np
from glob import glob
from torch.utils import data
from sklearn.model_selection import train_test_split
from skimage.transform import resize

sys.path.append("../")
sys.path.append("./")
from utils.base_util import readNiiImage, loadJson, saveNiiImage, getImageDirs


def resizeImg(img, coord, size=(128, 128, 128)):
    img_new = resize(img, size)
    coord_new = np.int16(np.asarray(coord) * size[0] / 481)
    return img_new, coord_new


class DatasetCoordsByCrop(data.Dataset):
    def __init__(self, images, coords):
        self.images = images
        self.coords = coords
        self.size = 256

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        image = self.images[index]
        coord = self.coords[index]
        coord = np.asarray(coord) - 112

        img = readNiiImage(image)
        img_crop = cropImageByCenter(img)
        img_norm = imgnorm(img_crop)

        img_tensor = torch.FloatTensor(img_norm).unsqueeze(0)
        coord_tensor = torch.Tensor(coord)

        image_size = [self.size, self.size, self.size]
        coord_tensor = (coord_tensor * 2 + 1) / torch.Tensor(image_size) - 1

        return img_tensor, coord_tensor.unsqueeze(0)


class DatasetCoordsByResize(data.Dataset):
    def __init__(self, images, coords, size):
        self.images = images
        self.coords = coords
        self.size = (size, size, size)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        image = self.images[index]
        coord = self.coords[index]
        coord = np.asarray(coord)

        img = readNiiImage(image)
        img_new, coord_new = resizeImg(img, coord, self.size)

        img_tensor = torch.FloatTensor(img_new).unsqueeze(0)
        coord_tensor = torch.Tensor(coord_new)

        coord_tensor = (coord_tensor * 2 + 1) / torch.Tensor(self.size) - 1

        return img_tensor, coord_tensor.unsqueeze(0)


def filterCoord(coord):
    if min(coord) < 112 or max(coord) > 360:
        return False
    return True


def getDataloader(root_dir, useB=True):
    # image_dirs = getImageDirs(root_dir)
    if useB:
        after_fix = "/*/img[AB].nii.gz"
    else:
        after_fix = "/*/imgA.nii.gz"
    image_dirs = glob(root_dir + after_fix)
    r = loadJson("part5/jsonfile/train_coordinate.json")

    files, coords = [], []

    for image in image_dirs:
        base1 = os.path.basename(image)[:4]
        base2 = re.findall("img\d+", image)[0]

        coord = r[base2][base1]
        # if filterCoord(coord):
        files.append(image)
        coords.append(coord[::-1])
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        coords,
        test_size=0.1,
        random_state=1,
    )

    params = {
        'batch_size': 3,
        'shuffle': True,
        'num_workers': 10,
        'worker_init_fn': np.random.seed(1)
    }
    size = 128
    train_set = DatasetCoordsByResize(X_train, y_train, size)
    train_dataloader = data.DataLoader(train_set, **params)
    test_set = DatasetCoordsByResize(X_test, y_test, size)
    test_dataloader = data.DataLoader(test_set, **params)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    root_dir = "/home/wangs/traindata"
    train_dataloader, test_dataloader = getDataloader(root_dir)
    for data, label in train_dataloader:
        print(label)
        break
