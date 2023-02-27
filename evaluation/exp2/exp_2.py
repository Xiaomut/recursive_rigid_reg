import os
import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("./")
from log import Log
from base_util import readNiiImage, resampleNiiImg, loadJson, saveJson
from metrics import dice, gd, ssim3d, ncc, gc
from model import Net
from exp2.image_util import cropImageByPoint, processOutPt


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def calMetrics(img1, img2, calCC=False, calGC=False, calGD=False):
    CC, GC, GD = None, None, None
    if calCC:
        CC = ncc.pearson_correlation(img1, img2).item()
    if calGC:
        GC = gc.GradientCorrelation3d()(img1, img2).item()
    if calGD:
        GD = gd.GradientDifference3d()(img1, img2).item()

    return CC, GC, GD


def updateCoord(num, r, limit):
    coord = r[num]["imgA"]
    coord_new = processOutPt(coord, (481, 481, 481), limit)
    return coord_new


def runSingle(num,
              base_path,
              filetype="0",
              calCC=True,
              calGC=True,
              calGD=False):
    # get images
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    imgA_gt = readNiiImage(os.path.join(base_path, "gt_imgA.nii.gz"))
    # warped_part3, elastix, sift
    if filetype == "0":
        imgB = readNiiImage(os.path.join(base_path, "elastix.nii.gz"))
        imgB_gt = readNiiImage(os.path.join(base_path, "gt_elastix.nii.gz"))
    elif filetype == "1":
        imgB = readNiiImage(os.path.join(base_path, "sift.nii.gz"))
        imgB_gt = readNiiImage(os.path.join(base_path, "gt_sift.nii.gz"))
    elif filetype == "2":
        imgB = readNiiImage(os.path.join(base_path, "warped_part3.nii.gz"))
        imgB_gt = readNiiImage(os.path.join(base_path, "gt_imgB.nii.gz"))

    coord_new = updateCoord(num, r, limit)
    imgA_crop = cropImageByPoint(imgA, coord_new, limit)
    imgB_crop = cropImageByPoint(imgB, coord_new, limit)

    return m_cc, m_gc, m_gd


def initJson(json_path):
    result = {
        "elastix": {
            "cc": [],
            "gc": [],
            "gd": []
        },
        "sift": {
            "cc": [],
            "gc": [],
            "gd": []
        },
        "warp": {
            "cc": [],
            "gc": [],
            "gd": []
        }
    }
    saveJson(result, json_path)
    return result


def finalCal():
    if not os.path.exists(json_path):
        result = initJson(json_path)
    else:
        result = loadJson(json_path)
    calCC, calGC, calGD = True, True, False

    errors = []

    for num in range(1, 45):
        try:
            base_path = os.path.join(data_path, f"img{num}")
            m_cc1, m_gc1, m_gd1 = runSingle(num, base_path, "0", calCC, calGC,
                                            calGD)
            m_cc2, m_gc2, m_gd2 = runSingle(num, base_path, "1", calCC, calGC,
                                            calGD)
            m_cc3, m_gc3, m_gd3 = runSingle(num, base_path, "2", calCC, calGC,
                                            calGD)
            result["elastix"]["cc"].append(m_cc1)
            result["elastix"]["gc"].append(m_gc1)
            result["elastix"]["gd"].append(m_gd1)
            result["sift"]["cc"].append(m_cc2)
            result["sift"]["gc"].append(m_gc2)
            result["sift"]["gd"].append(m_gd2)
            result["warp"]["cc"].append(m_cc3)
            result["warp"]["gc"].append(m_gc3)
            result["warp"]["gd"].append(m_gd3)
        except Exception as e:
            errors.append(num)
            log.info(f"--- file {num} has something wrong {e} ---")


if __name__ == "__main__":

    data_path = "Y:/testdata/"
    json_path = "exp2/exp2.json"

    log = Log(filename="exp2/log/exp2.log").getlog()

    r = loadJson("part5/jsonfile/test_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]

    finalCal()
