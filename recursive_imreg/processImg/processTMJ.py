import os
import sys
import numpy as np
from multiprocessing.dummy import Pool

sys.path.append('../')
sys.path.append('./')
from utils import base_util, image_util, histmatching


def getCropHis():
    base_dir = "Y:/traindata2"
    infos = base_util.loadJson(file="files/img_infos.json")

    for i in range(1, 124):
        img_dir = os.path.join(base_dir, f"img{i}")

        imgA, infos = base_util.readNiiImage(
            os.path.join(img_dir, "imgA_crop_his.nii.gz"), True)
        gt_imgA = base_util.readNiiImage(
            os.path.join(img_dir, "gt_imgA_crop.nii.gz"))
        imgB = base_util.readNiiImage(
            os.path.join(img_dir, "imgB_crop_his.nii.gz"))
        gt_imgB = base_util.readNiiImage(
            os.path.join(img_dir, "gt_imgB_crop.nii.gz"))

        imgA_tmj = imgA * gt_imgA
        imgB_tmj = imgB * gt_imgB

        base_util.saveNiiImage(
            imgA_tmj, infos, os.path.join(img_dir,
                                          "imgA_crop_his_only.nii.gz"))
        base_util.saveNiiImage(
            imgB_tmj, infos, os.path.join(img_dir,
                                          "imgB_crop_his_only.nii.gz"))
        # break


if __name__ == "__main__":
    getCropHis()