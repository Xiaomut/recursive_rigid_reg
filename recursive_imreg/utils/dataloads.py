import os
import sys
import torch
import numpy as np
from torch.utils import data
from functools import lru_cache

from multiprocessing.dummy import Pool

sys.path.append("../")
sys.path.append("./")
from utils.base_util import getImageDirs, loadJson, readNiiImage, saveNiiImage
from utils.image_util import cropImageByPoint, processOutPt
from utils.histmatching import matching
from config import Config as conf


class DatasetPtCrop(data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # imgA_crop, imgB_crop, filename = self.datas[index]
        imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename = self.datas[
            index]

        imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0)
        imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0)

        imgA_gt_tensor = torch.FloatTensor(imgA_gt_crop).unsqueeze(0)
        imgB_gt_tensor = torch.FloatTensor(imgB_gt_crop).unsqueeze(0)

        # return imgA_tensor, imgB_tensor
        return imgA_tensor, imgB_tensor, imgA_gt_tensor, imgB_gt_tensor, filename


def processData(file, r, limit):
    """only get dot image"""
    base_dir = os.path.expanduser(file)
    filenum = os.path.basename(base_dir)
    imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    imgA_gt = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB_gt = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

    # 更新一下坐标, 防止溢出
    coorA = r[filenum]["imgA"]
    coorA = processOutPt(coorA, imgA.shape, limit)
    coorB = r[filenum]["imgB"]
    coorB = processOutPt(coorB, imgB.shape, limit)

    imgA_input = imgA_gt * imgA
    imgB_input = imgB_gt * imgB
    # histmatch  B -> A
    imgB_input = matching(imgA_input, imgB_input)

    imgA_crop = cropImageByPoint(imgA_input, coorA, limit)
    imgB_crop = cropImageByPoint(imgB_input, coorB, limit)
    return imgA_crop, imgB_crop, os.path.basename(base_dir)


def processMulData(file, r, limit):
    """get four images"""
    base_dir = os.path.expanduser(file)
    filenum = os.path.basename(base_dir)
    imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    imgA_gt = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB_gt = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

    # 更新一下坐标, 防止溢出
    coorA = r[filenum]["imgA"]
    coorA = processOutPt(coorA, imgA.shape, limit)
    coorB = r[filenum]["imgB"]
    coorB = processOutPt(coorB, imgB.shape, limit)

    # imgA_input = imgA_gt * imgA
    # imgB_input = imgB_gt * imgB
    # histmatch  B -> A
    # imgB_input = matching(imgA_input, imgB_input)

    imgA_crop = cropImageByPoint(imgA, coorA, limit)
    imgB_crop = cropImageByPoint(imgB, coorB, limit)
    imgA_gt_crop = cropImageByPoint(imgA_gt, coorA, limit)
    imgB_gt_crop = cropImageByPoint(imgB_gt, coorB, limit)

    filename = os.path.basename(base_dir)
    return imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename


def getDatas(root_dir, pool_size=12):
    train_datas = []
    test_datas = []

    image_dirs = getImageDirs(root_dir)

    if "train" in root_dir:
        r = loadJson("files/train_coordinate.json")
        pre_images, suf_images = image_dirs[:147] + image_dirs[
            153:], image_dirs[147:153]
        X_train_imgs = suf_images[:2]
        X_test_imgs = suf_images[2:]
    elif "test" in root_dir:
        r = loadJson("files/test_coordinate.json")
        X_train_imgs = image_dirs[:36]
        X_test_imgs = image_dirs[36:]
    else:
        raise NotImplementedError(f"The file {root_dir} doesn't exist!")
    limit = [256, -60, 200, 128, 128, 128]

    pool = Pool(pool_size)
    train_res = []
    for i in range(len(X_train_imgs)):
        train_temp = pool.apply_async(processMulData,
                                      (X_train_imgs[i], r, limit))
        train_res.append(train_temp)
    pool.close()
    pool.join()

    for train_r in train_res:
        train_datas.append(train_r.get())
    #     imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename = processMulData(
    #         X_train_imgs[i], r, limit)
    #     train_datas.append(
    #         (imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename))

    pool = Pool(pool_size)
    test_res = []
    for i in range(len(X_test_imgs)):
        test_temp = pool.apply_async(processMulData,
                                     (X_test_imgs[i], r, limit))
        test_res.append(test_temp)
    pool.close()
    pool.join()

    for test_r in test_res:
        test_datas.append(test_r.get())
    #     imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename = processMulData(
    #         X_test_imgs[i], r, limit)
    #     test_datas.append(
    #         (imgA_crop, imgB_crop, imgA_gt_crop, imgB_gt_crop, filename))
    return train_datas, test_datas


def getDataloader(root_dir):
    """ pt_mode: `norch` or `condyle` """

    X_train, X_test = getDatas(root_dir)
    params_train = {
        'batch_size': conf.batch_size,
        'shuffle': False,
        'num_workers': conf.numwork,
        'worker_init_fn': np.random.seed(1)
    }
    train_set = DatasetPtCrop(X_train)
    train_dataloader = data.DataLoader(train_set, **params_train)
    test_set = DatasetPtCrop(X_test)
    test_dataloader = data.DataLoader(test_set, **params_train)
    return train_dataloader, test_dataloader
    """none"""


if __name__ == "__main__":
    a, b = getDataloader(os.path.join(conf.root_path, "testdata"))
