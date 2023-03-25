import os
import sys
import torch
import random
import numpy as np
from torch.utils import data

sys.path.append("../")
sys.path.append("./")
from utils.base_util import getImageDirs, loadJson, readNiiImage, saveNiiImage
from utils.image_util import cropImageByPoint, processOutPt
from utils.histmatching import matching, histForTrain
from config import Config as conf


class DatasetPtCrop(data.Dataset):
    def __init__(self, image_dirs, r, limit, norm=False):
        self.image_dirs = image_dirs
        self.r = r
        self.limit = limit
        # self.t_quantiles, self.t_values = self.getBaseInfo()

    def __len__(self):
        return len(self.image_dirs)

    def getBaseInfo(self, file="/home/wangs/base_imgs/base.nii.gz"):
        template = readNiiImage(file)
        nt_data_array = template.ravel()
        t_values, t_counts = np.unique(nt_data_array, return_counts=True)
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]
        return t_quantiles, t_values

    def __getitem__(self, index):
        base_dir = os.path.expanduser(self.image_dirs[index])
        filenum = os.path.basename(base_dir)
        imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))  # _his
        imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
        # imgA = histForTrain(self.t_quantiles, self.t_values, imgA)
        # imgB = histForTrain(self.t_quantiles, self.t_values, imgB)
        imgA_gt = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
        imgB_gt = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

        # 更新一下坐标, 防止溢出
        coorA = self.r[filenum]["imgA"]
        coorA = processOutPt(coorA, imgA.shape, self.limit)
        coorB = self.r[filenum]["imgB"]
        coorB = processOutPt(coorB, imgB.shape, self.limit)

        imgA_crop = cropImageByPoint(imgA, coorA, self.limit)
        imgB_crop = cropImageByPoint(imgB, coorB, self.limit)
        # histequal
        # imgA_crop = histForTrain(self.t_quantiles, self.t_values, imgA_crop)
        # imgB_crop = matching(imgA_crop, imgB_crop)

        imgA_gt_crop = cropImageByPoint(imgA_gt, coorA, self.limit)
        imgB_gt_crop = cropImageByPoint(imgB_gt, coorB, self.limit)

        imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0)
        imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0)

        imgA_gt_tensor = torch.FloatTensor(imgA_gt_crop).unsqueeze(0)
        imgB_gt_tensor = torch.FloatTensor(imgB_gt_crop).unsqueeze(0)

        return imgA_tensor, imgB_tensor, imgA_gt_tensor, imgB_gt_tensor, os.path.basename(
            base_dir)


def getDataloader(root_dir):
    """ pt_mode: `norch` or `condyle` """
    image_dirs = getImageDirs(root_dir)

    for i in [54, 64]:  # , 13, 41, 43, 67, 73, 112, 113
        image_dirs.remove(os.path.join(root_dir, f"img{i}"))

    random.shuffle(image_dirs)
    ratio = int(0.8 * len(image_dirs))
    X_train = image_dirs[:ratio]
    X_test = image_dirs[ratio:]

    r = loadJson("files/train_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]
    # limit = [200, -60, 196, 128, 128, 128]

    params_train = {
        'batch_size': conf.batch_size,
        'shuffle': True,
        'num_workers': conf.numwork,
        'worker_init_fn': np.random.seed(1)
    }
    train_set = DatasetPtCrop(X_train, r=r, limit=limit)
    train_dataloader = data.DataLoader(train_set, **params_train)
    test_set = DatasetPtCrop(X_test, r=r, limit=limit)
    test_dataloader = data.DataLoader(test_set, **params_train)
    return train_dataloader, test_dataloader


def getTestloader(root_dir="/home/wangs/testdata"):
    image_dirs = getImageDirs(root_dir)

    r = loadJson("files/test_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]
    # limit = [200, -60, 196, 128, 128, 128]

    params_test = {
        'batch_size': conf.batch_size,
        'shuffle': False,
        'num_workers': conf.numwork,
        'worker_init_fn': np.random.seed(1)
    }
    test_set = DatasetPtCrop(image_dirs, r=r, limit=limit)
    test_dataloader = data.DataLoader(test_set, **params_test)
    return test_dataloader


if __name__ == "__main__":
    a, b = getDataloader(os.path.join(conf.root_path, "traindata2"))
    for i in a:
        print(i.shape)
