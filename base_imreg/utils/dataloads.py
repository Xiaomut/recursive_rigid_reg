import os
import sys
import numpy as np
import random
from torch.utils import data

sys.path.append("../")
sys.path.append("./")
from utils.base_util import getImageDirs, loadJson, readNiiImage, saveNiiImage
from utils.image_util import cropImageByCenter, invMatrix, decomposeMatrixDegree, imgnorm, reviseMtxToCrop
from config import Config as conf

class DatasetCenterCrop(data.Dataset):
    def __init__(self,
                 image_dirs,
                 crop_size,
                 norm=True,
                 change=True,
                 mode="B2A"):
        self.image_dirs = image_dirs
        self.norm = norm
        self.change = change
        self.mode = mode
        self.crop_size = crop_size
        self.shape = (crop_size, crop_size, crop_size)

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        base_dir = os.path.expanduser(self.image_dirs[index])
        imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
        imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
        label = np.loadtxt(os.path.join(base_dir, "label.txt"))
        label_revise = reviseMtxToCrop(imgA.shape, self.shape, label)

        if self.mode == "A2B":
            label_revise = decomposeMatrixDegree(label_revise,
                                                 change=self.change)  # A2B
        elif self.mode == "B2A":
            label_revise = invMatrix(label_revise, change=self.change)  # B2A

        imgA_crop = cropImageByCenter(imgA, self.crop_size)
        imgB_crop = cropImageByCenter(imgB, self.crop_size)

        imgA_crop = gaussNoise(imgA_crop)
        imgB_crop = gaussNoise(imgB_crop)
        if self.norm:
            imgA_crop = imgnorm(imgA_crop)
            imgB_crop = imgnorm(imgB_crop)

        imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0)
        imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0)
        label_revise_tensor = torch.FloatTensor(label_revise).view((6, ))

        return imgA_tensor, imgB_tensor, label_revise_tensor, os.path.basename(
            base_dir)


def getDataloader(root_dir, size, mode):
    image_dirs = getImageDirs(root_dir)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     image_dirs,
    #     list(range(len(image_dirs))),
    #     test_size=conf.testsize,
    #     random_state=1,
    # )

    random.shuffle(image_dirs)
    ratio = int(0.8 * len(image_dirs))
    X_train = image_dirs[:ratio]
    X_test = image_dirs[ratio:]

    params = {
        'batch_size': conf.batch_size,
        'shuffle': True,
        'num_workers': conf.numwork,
        'worker_init_fn': np.random.seed(1)
    }
    train_set = DatasetCenterCrop(X_train,
                                  crop_size=size,
                                  norm=conf.norm,
                                  change=conf.change,
                                  mode=mode)
    train_dataloader = data.DataLoader(train_set, **params)
    test_set = DatasetCenterCrop(X_test,
                                 crop_size=size,
                                 norm=conf.norm,
                                 change=conf.change,
                                 mode=mode)
    test_dataloader = data.DataLoader(test_set, **params)
    return train_dataloader, test_dataloader


def getTestloader(root_dir, size, mode):
    image_dirs = getImageDirs(root_dir)

    params_test = {
        'batch_size': conf.batch_size,
        'shuffle': False,
        'num_workers': conf.numwork,
        'worker_init_fn': np.random.seed(1)
    }
    test_set = DatasetCenterCrop(image_dirs, 
                                crop_size=size,
                                 norm=conf.norm,
                                 change=conf.change,
                                 mode=mode)
    test_dataloader = data.DataLoader(test_set, **params_test)
    return test_dataloader