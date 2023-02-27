import os
import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("./")
from log import Log
from utils.base_util import readNiiImage, resampleNiiImg, loadJson, saveJson
from metrics import dice, gd, ssim3d, ncc, gc
from model import Net
from exp1.image_util import imgnorm, cropImageByCenter, reviseMtxFromCrop, composeMatrixFromDegree


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def loadModel(model_path):
    """ load the net """
    net = Net(8, 4)
    net.load_state_dict(
        torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()
    return net


def warp(imgA, imgB, net, ifsave=False):
    """ 
    warp the image by net's output 
    
    @params: `imgA` and `imgB` must be 3D-ndarray
    @return: `imgA_crop`, `imgA_input`, `warped_img_crop`, `warped_img`. All is tensor
    """

    midshape = (256, 256, 256)

    imgA_crop = imgnorm(cropImageByCenter(imgA, size=256))
    imgB_crop = imgnorm(cropImageByCenter(imgB, size=256))
    imgA_input = imgToTensor(imgA_crop)
    imgB_input = imgToTensor(imgB_crop)
    pred_degree = net(imgA_input, imgB_input)

    fake_mtx = composeMatrixFromDegree(pred_degree.detach().numpy())
    warped_img_crop = resampleNiiImg(fake_mtx, imgB_input)

    pred_mtx_revise = reviseMtxFromCrop(imgA.shape, midshape, fake_mtx)
    imgB_tensor = imgToTensor(imgB)

    infos = loadJson("files/img_infos.json")
    if ifsave:
        # save_name = os.path.join(base_path, resample_name)
        warped_img = resampleNiiImg(pred_mtx_revise, imgB_tensor, infos,
                                    resample_name)
    else:
        warped_img = resampleNiiImg(pred_mtx_revise, imgB_tensor, infos)
    return imgA_crop, imgA_input, warped_img_crop, warped_img


def calMetrics(img1, img2, calCC=False, calGC=False, calGD=False):
    CC, GC, GD = None, None, None
    if calCC:
        CC = ncc.pearson_correlation(img1, img2).item()
    if calGC:
        GC = gc.GradientCorrelation3d()(img1, img2).item()
    if calGD:
        GD = gd.GradientDifference3d()(img1, img2).item()

    return CC, GC, GD


def runAll(num, base_path, saveImg=False, calCC=True, calGC=True, calGD=False):
    # get model
    net = loadModel(model_path)
    # get images
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_path, "imgB.nii.gz"))
    # get warped image. [ndarray, tensor, tensor, tensor]
    imgA_crop, imgA_input, warped_img_crop, warped_img = warp(
        imgA, imgB, net, saveImg)
    m_cc, m_gc, m_gd = calMetrics(imgA_input, warped_img_crop, calCC, calGC,
                                  calGD)
    print(m_cc, m_gc, m_gd)


def runSingle(num,
              base_path,
              filetype="0",
              calCC=True,
              calGC=True,
              calGD=False):
    # get images
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    # warped_part3, elastix, sift
    if filetype == "0":
        imgB = readNiiImage(os.path.join(base_path, "elastix.nii.gz"))
    elif filetype == "1":
        imgB = readNiiImage(os.path.join(base_path, "sift.nii.gz"))
    elif filetype == "2":
        imgB = readNiiImage(os.path.join(base_path, "warped_part3.nii.gz"))

    # # full image
    # imgA_tensor = imgToTensor(imgA)
    # imgB_tensor = imgToTensor(imgB)
    # m_cc, m_gc, m_gd = calMetrics(imgA_tensor, imgB_tensor, True, False, False)

    # crop image
    imgA_crop = imgToTensor(cropImageByCenter(imgA))
    imgB_crop = imgToTensor(cropImageByCenter(imgB))
    m_cc, m_gc, m_gd = calMetrics(imgA_crop, imgB_crop, calCC, calGC, calGD)
    log.info(
        f"[file {num} Type {filetype}] -> [cc: {m_cc:.4f}] -> [gc: {m_gc:.4f}]"
    )
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
    model_path = "X:/Codes/base_imreg/result/pth1632/result_B2A.pth"
    resample_name = "exp1.nii.gz"
    json_path = "exp1/exp1.json"

    log = Log(filename="exp1/log/exp1.log").getlog()

    # runAll(num)
    # runSingle(num)
    finalCal()
