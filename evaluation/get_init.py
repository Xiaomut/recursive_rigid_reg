import os
import torch
import numpy as np

from log import Log
from base_util import readNiiImage, saveNiiImage, resampleNiiImg, loadJson, saveJson
from metrics import dice, gd, ssim3d, ncc, gc
from exp3.image_util import cropImageByPoint, processOutPt, reviseMtxFromCrop, composeMatrixFromDegree, cropImageByPointTest


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def calMetrics(img1,
               img2,
               img3=None,
               img4=None,
               calCC=False,
               calGC=False,
               calGD=False,
               calDice=False):
    CC, GC, GD, DICE = 0, 0, 0, 0
    if calCC:
        CC = ncc.pearson_correlation(img1, img2).item()
    if calGC:
        GC = gc.GradientCorrelation3d()(img1, img2).item()
    if calGD:
        GD = gd.GradientDifference3d()(img1, img2).item()
    if calDice and img3 is not None:
        DICE = dice.dice_coeff(img3, img4)

    return CC, GC, GD, DICE


def updateCoord(num, r, limit):
    coord = r[num]["imgA"]
    coord_new = processOutPt(coord, (481, 481, 481), limit)
    return coord_new


def getCoord(num):
    shape = (481, 481, 481)
    # 更新一下坐标, 防止溢出
    coorA = r[f"img{num}"]["imgA"]
    coorA = processOutPt(coorA, shape, limit)
    coorB = r[f"img{num}"]["imgB"]
    coorB = processOutPt(coorB, shape, limit)

    return coorA, coorB


def getImgs(num, base_path):
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_path, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_path, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_path, "gt_imgB.nii.gz"))

    return imgA, gt_imgA, imgB, gt_imgB


def runSingle(num):
    base_path = os.path.join(data_path, f"img{num}")

    # get fixed images
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_path, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_path, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_path, "gt_imgB.nii.gz"))

    coorA, _ = getCoord(num)
    # crop image and get TMJ
    imgA_input = imgA * gt_imgA
    imgB_input = imgB * gt_imgB
    imgA_crop = imgToTensor(cropImageByPointTest(imgA_input, coorA, limit))
    gt_imgA_crop = imgToTensor(cropImageByPointTest(gt_imgA, coorA, limit))
    imgB_crop = imgToTensor(cropImageByPointTest(imgB_input, coorA, limit))
    gt_imgB_crop = imgToTensor(cropImageByPointTest(gt_imgB, coorA, limit))

    m_cc, m_gc, m_gd, m_dice = calMetrics(imgA_crop, imgB_crop, gt_imgA_crop,
                                          gt_imgB_crop, calCC, calGC, calGD,
                                          calDice)
    log.info(
        f"[file {num} Type m ] -> [cc: {m_cc:.4f}] -> [gc: {m_gc:.4f}] -> [gd: {m_gd:.4f}] -> [dice: {m_dice:.4f}]"
    )
    return m_cc, m_gc, m_gd, m_dice


if __name__ == "__main__":
    data_path = "Y:/testdata/"
    r = loadJson("files/test_coordinate.json")
    infos = loadJson("files/img_infos.json")
    limit = [200, -60, 128, 128, 128, 128]
    calCC, calGC, calGD, calDice = True, True, True, True

    log_name = "files/init.log"
    log = Log(filename=log_name, mode="w").getlog()

    for num in range(1, 45):
        m_cc3, m_gc3, m_gd3, m_dice3 = runSingle(num)