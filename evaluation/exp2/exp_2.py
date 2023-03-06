import os
import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("./")
from log import Log
from base_util import readNiiImage, resampleNiiImg, loadJson, saveJson
from metrics import dice, gd, ssim3d, ncc, gc
from exp2.model import Net
from exp2.image_util import cropImageByPoint, processOutPt, imgnorm, reviseMtxFromCrop, composeMatrixFromDegree


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def loadModel(model_path):
    """ load the net """
    net = Net(8, 4)
    net_params = torch.load(model_path,
                            map_location=torch.device('cpu'))["net"]
    net.load_state_dict(net_params)
    net.eval()
    return net


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


def getImgs(num, base_dir):
    imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

    return imgA, gt_imgA, imgB, gt_imgB


def getAddTheta(coorA, coorB):
    t = np.array([
        coorB[0] - coorA[0], coorB[1] - coorA[1], coorB[2] - coorA[2]
    ]) / 481 * -2
    theta_add = np.array([
        [0, 0, 0, t[0]],
        [0, 0, 0, t[1]],
        [0, 0, 0, t[2]],
    ])
    return torch.FloatTensor(theta_add * -1)


def warp(num, net, ifsave=False):

    base_path = os.path.join(data_path, f"img{num}")
    coorA, coorB = getCoord(num)
    imgA, gt_imgA, imgB, gt_imgB = getImgs(num, base_path)

    imgA_input = imgA * gt_imgA
    imgB_input = imgB * gt_imgB
    imgA_crop = cropImageByPoint(imgA_input, coorA, limit)
    imgB_crop = cropImageByPoint(imgB_input, coorB, limit)

    # imgA_crop = imgnorm(imgA_crop)
    # imgB_crop = imgnorm(imgB_crop)

    imgA_tensor = imgToTensor(imgA_crop)
    imgB_tensor = imgToTensor(imgB_crop)

    pred_degree = net(imgA_tensor, imgB_tensor)
    fake_mtx = composeMatrixFromDegree(pred_degree.detach().numpy(),
                                       change=True)
    pred_mtx_revise = reviseMtxFromCrop(imgA.shape, imgA_crop.shape, fake_mtx)

    # get theta_add and revise the matrix, `fake theta` is the final matrix
    theta_add = getAddTheta(coorA, coorB)
    fake_theta = torch.add(theta_add, pred_mtx_revise).type(torch.float32)

    infos = loadJson("files/img_infos.json")
    ori_save_name = os.path.join(base_path, resample_name)
    seg_name = ori_save_name.replace("_now", "_seg")

    img_o_tensor = imgToTensor(imgB)
    img_gt_tensor = imgToTensor(gt_imgB)
    if ifsave:
        _, warped_img = resampleNiiImg(fake_theta, img_o_tensor, infos,
                                       ori_save_name)
        _, warped_img_gt = resampleNiiImg(fake_theta, img_gt_tensor, infos,
                                          seg_name, "nearest")
        log.info(f"save file: {ori_save_name} done!")
    else:
        _, warped_img = resampleNiiImg(fake_theta, img_o_tensor)
        _, warped_img_gt = resampleNiiImg(fake_theta,
                                          img_gt_tensor,
                                          mode="nearest")
    return imgA_crop, imgA_input, warped_img, warped_img_gt


def runSingle(num,
              filetype="0",
              calCC=True,
              calGC=True,
              calGD=True,
              calDice=True,
              onlySave=False):
    base_path = os.path.join(data_path, f"img{num}")

    # get fixed images
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_path, "gt_imgA.nii.gz"))
    if filetype == "0":
        imgB = readNiiImage(os.path.join(base_path, "elastix.nii.gz"))
        gt_imgB = readNiiImage(os.path.join(base_path, "gt_elstix.nii.gz"))
    elif filetype == "1":
        imgB = readNiiImage(os.path.join(base_path, "sift.nii.gz"))
        gt_imgB = readNiiImage(os.path.join(base_path, "gt_sift.nii.gz"))
    elif filetype == "2":
        net = loadModel(model_path)
        _, _, imgB, gt_imgB = warp(num, net, save_photo)

    if onlySave:
        exit(0)

    coorA, _ = getCoord(num)
    # crop image and get TMJ
    imgA_input = imgA * gt_imgA
    imgB_input = imgB * gt_imgB
    imgA_crop = imgToTensor(cropImageByPoint(imgA_input, coorA, limit))
    gt_imgA_crop = imgToTensor(cropImageByPoint(gt_imgA, coorA, limit))
    imgB_crop = imgToTensor(cropImageByPoint(imgB_input, coorA, limit))
    gt_imgB_crop = imgToTensor(cropImageByPoint(gt_imgB, coorA, limit))

    m_cc, m_gc, m_gd, m_dice = calMetrics(imgA_crop, imgB_crop, gt_imgA_crop,
                                          gt_imgB_crop, calCC, calGC, calGD,
                                          calDice)
    log.info(
        f"[file {num} Type {filetype}] -> [cc: {m_cc:.4f}] -> [gc: {m_gc:.4f}] -> [gd: {m_gd:.4f}] -> [dice: {m_dice:.4f}]"
    )
    return m_cc, m_gc, m_gd, m_dice


if __name__ == "__main__":

    data_path = "Y:/testdata/"
    r = loadJson("files/test_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]

    # need change before running codes
    save_photo = True  # decide if save the warped image
    onlySave = True  # decide if calculate the metrics
    model_name = "la2.5"  # la2.5  la2  la1
    json_path = "exp2/exp2.json"
    log_name = f"exp2/log/exp2_{model_name}.log"
    model_path = f"X:/Exp-New/part6_3_seg/result/{model_name}/best_part6_3_seg_B2A.pth"
    resample_name = f"exp2_{model_name}_now.nii.gz"  # _la2.5  _la2  _la1

    if save_photo:
        log = Log(filename="exp2/log/save_photo.log", mode="a").getlog()
    else:
        log = Log(filename=log_name, mode="w").getlog()

    # only for saving photo. Remember to change the param save_photo
    single_num = 4
    _, _, _, _ = runSingle(single_num, "2", False, False, False, False,
                           save_photo)
    exit(0)

    # !!! Attention !!! type `0` and `1` only run once
    for num in range(1, 45):
        calCC, calGC, calGD, calDice = True, True, True, True
        # m_cc1, m_gc1, m_gd1, m_dice1 = runSingle(num, "0", calCC, calGC, calGD,
        #                                          calDice)
        # m_cc2, m_gc2, m_gd2, m_dice2 = runSingle(num, "1", calCC, calGC, calGD,
        #                                          calDice)
        m_cc3, m_gc3, m_gd3, m_dice3 = runSingle(num, "2", calCC, calGC, calGD,
                                                 calDice, onlySave)
