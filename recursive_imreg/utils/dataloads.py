import os
import sys
import torch
import numpy as np
from torch.utils import data

sys.path.append("../")
sys.path.append("./")
from utils.base_util import getImageDirs, loadJson, readNiiImage, saveNiiImage
from utils.image_util import cropImageByPoint, processOutPt
from utils.histmatching import matching
from config import Config as conf


class DatasetPtCrop(data.Dataset):
    def __init__(self, image_dirs, r, limit, norm=False):
        self.image_dirs = image_dirs
        self.r = r
        self.limit = limit

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        base_dir = os.path.expanduser(self.image_dirs[index])
        filenum = os.path.basename(base_dir)
        imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
        imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
        imgA_gt = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
        imgB_gt = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

        # 更新一下坐标, 防止溢出
        coorA = self.r[filenum]["imgA"]
        coorA = processOutPt(coorA, imgA.shape, self.limit)
        coorB = self.r[filenum]["imgB"]
        coorB = processOutPt(coorB, imgB.shape, self.limit)

        imgA_crop = cropImageByPoint(imgA, coorA, self.limit)
        imgB_crop = cropImageByPoint(imgB, coorB, self.limit)
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
    image_dirs.remove(os.path.join(root_dir, "img54"))
    image_dirs.remove(os.path.join(root_dir, "img64"))

    ratio = int(0.8 * len(image_dirs))
    X_train = image_dirs[:ratio]
    X_test = image_dirs[ratio:]

    r = loadJson("files/train_coordinate.json")
    # limit = [256, -60, 200, 128, 128, 128]
    limit = [220, -80, 128, 128, 128, 128]

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
