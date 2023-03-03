import os
import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("./")
from log import Log
from base_util import readNiiImage, saveNiiImage, resampleNiiImg, loadJson, saveJson
from metrics import dice, gd, ssim3d, ncc, gc
from exp3.image_util import cropImageByPoint, processOutPt, reviseMtxFromCrop, composeMatrixFromDegree, cropImageByPointTest
from exp3.models import recurnet_4img, recurnet_cbct, recurnet_cbct_pro


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def loadModel(model_path, n):
    """ load the net """
    s = torch.load(model_path, map_location="cpu")
    device = torch.device('cpu')
    if "pro" in model_path:
        if "corr_pro" in model_path:
            model_type = "0"
        else:
            model_type = "1"
        net = recurnet_cbct_pro.RecursiveCascadeNetwork(
            n, 8, 32, device, False, False, model_type, s, True)
    elif "corr" in model_path:
        net = recurnet_cbct.RecursiveCascadeNetwork(n, 8, 32, device, False,
                                                    False, s, True)
    else:
        net = recurnet_4img.RecursiveCascadeNetwork(n, 8, device, s, True)
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


def getImgs(num, base_path):
    imgA = readNiiImage(os.path.join(base_path, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_path, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_path, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_path, "gt_imgB.nii.gz"))

    return imgA, gt_imgA, imgB, gt_imgB


def getOriThetas(rts, coorA, coorB, shape1, shape2):
    t = np.array([
        coorB[0] - coorA[0], coorB[1] - coorA[1], coorB[2] - coorA[2]
    ]) / 481 * -2
    theta_add = np.array([
        [0, 0, 0, t[0]],
        [0, 0, 0, t[1]],
        [0, 0, 0, t[2]],
    ])

    rts = [theta.detach() for theta in rts]

    mtxs = []  # save crop thetas
    for rt in rts:
        mtxs.append(composeMatrixFromDegree(rt))

    mtx_revises = []
    for mtx in mtxs:
        mtx_revises.append(reviseMtxFromCrop(shape1, shape2, mtx))

    mtx_revises[0] = torch.add(torch.tensor(theta_add * -1),
                               mtx_revises[0]).type(torch.float32)

    return mtx_revises


def warp(num, net, ifsave=False):

    base_path = os.path.join(data_path, f"img{num}")
    coorA, coorB = getCoord(num)
    imgA, gt_imgA, imgB, gt_imgB = getImgs(num, base_path)

    # 截取原图像
    imgA_crop = cropImageByPoint(imgA, coorA, limit)
    imgB_crop = cropImageByPoint(imgB, coorB, limit)

    # 截取分割图像并保存
    imgA_gt_crop = cropImageByPoint(gt_imgA, coorA, limit)
    imgB_gt_crop = cropImageByPoint(gt_imgB, coorB, limit)

    imgA_tensor = imgToTensor(imgA_crop)
    imgB_tensor = imgToTensor(imgB_crop)
    img_o_tensor = imgToTensor(imgB)
    imgA_gt_tensor = imgToTensor(imgA_gt_crop)
    imgB_gt_tensor = imgToTensor(imgB_gt_crop)
    img_gt_tensor = imgToTensor(gt_imgB)

    results = net(imgA_tensor, imgB_tensor, imgA_gt_tensor, imgB_gt_tensor)
    if len(results) == 3:
        stem_results, thetas, stem_results_gt = results
    if len(results) == 5:
        stem_results, thetas, stem_results_gt, feaA_pro, feaB_pro = results

    save_name = os.path.join(base_path, resample_name)

    # get final warped
    thetas_ori = getOriThetas(thetas, coorA, coorB, imgA.shape,
                              imgB_crop.shape)
    warped = img_o_tensor
    warped_gt = img_gt_tensor
    for i, (stem_result, theta_o, stem_result_gt) in enumerate(
            zip(stem_results, thetas_ori, stem_results_gt)):
        # 需要将其相乘进行存储
        warped_crop = (stem_result *
                       stem_result_gt).squeeze(0).squeeze(0).cpu().detach()
        # 保存被一个阶段的图像
        # saveNiiImage(warped_gt.numpy(), infos, save_name.replace(".nii.gz", f"_crop_{nums}{i+1}.nii.gz"))
        # warped = resampleNiiImg(theta_o, warped, infos, save_name.replace(".nii.gz", f"_{nums}{i+1}.nii.gz"))
        warped, warped_numpy = resampleNiiImg(theta_o, warped)
        warped_gt, warped_gt_numpy = resampleNiiImg(theta_o,
                                                    warped_gt,
                                                    mode="nearest")
    if ifsave:
        # 只保存最后的图像
        warped = warped.type(torch.ShortTensor).squeeze().squeeze().numpy()
        saveNiiImage(warped_crop.numpy(), infos,
                     save_name.replace(".nii.gz", "_crop_final.nii.gz"))
        saveNiiImage(warped, infos,
                     save_name.replace(".nii.gz", "_final.nii.gz"))
    return imgA, imgA_crop, warped_numpy, warped_gt_numpy


def runSingle(num,
              n,
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
    imgB, gt_imgB = None, None
    if filetype == "0":
        imgB = readNiiImage(os.path.join(base_path, "elastix.nii.gz"))
        gt_imgB = readNiiImage(os.path.join(base_path, "gt_elstix.nii.gz"))
    elif filetype == "1":
        imgB = readNiiImage(os.path.join(base_path, "sift.nii.gz"))
        gt_imgB = readNiiImage(os.path.join(base_path, "gt_sift.nii.gz"))
    elif filetype == "2":
        net = loadModel(model_path, n)
        _, _, imgB, gt_imgB = warp(num, net, save_photo)

    if onlySave and filetype == "2":
        exit(0)

    coorA, _ = getCoord(num)
    # crop image and get TMJ
    imgA_input = imgA  # * gt_imgA
    imgB_input = imgB  # * gt_imgB
    imgA_crop = imgToTensor(cropImageByPointTest(imgA_input, coorA, limit))
    gt_imgA_crop = imgToTensor(cropImageByPointTest(gt_imgA, coorA, limit))
    imgB_crop = imgToTensor(cropImageByPointTest(imgB_input, coorA, limit))
    gt_imgB_crop = imgToTensor(cropImageByPointTest(gt_imgB, coorA, limit))

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
    infos = loadJson("files/img_infos.json")
    limit = [200, -60, 128, 128, 128, 128]

    # need change before running codes
    save_photo = False  # decide if save the warped image
    onlySave = False  # decide if calculate the metrics
    n = 2
    corr, pro = True, False
    pre = "_corr" if corr else ""
    post = "_pro" if pro else ""
    model_name = f"cas{n}{pre}{post}"  # la2.5  la2  la1
    json_path = "exp3/exp3.json"
    log_name = f"exp3/log/{model_name}.log"
    model_path = f"X:/Codes/recursive_imreg/recurse/cas{n}/cur{pre}{post}_0832/best_recurse.pth"
    resample_name = f"exp3_{model_name}_now.nii.gz"  # _la2.5  _la2  _la1

    log = Log(filename=log_name, mode="w").getlog()
    if save_photo == True:
        log = Log(filename="exp3/log/save_photo.log", mode="w").getlog()

    # !!! Attention !!! type `0` and `1` only run once
    for num in range(1, 45):
        calCC, calGC, calGD, calDice = True, True, True, True
        # m_cc1, m_gc1, m_gd1, m_dice1 = runSingle(num, n, "0", calCC, calGC, calGD,
        #                                          calDice)
        # m_cc2, m_gc2, m_gd2, m_dice2 = runSingle(num, n, "1", calCC, calGC, calGD,
        #                                          calDice)
        m_cc3, m_gc3, m_gd3, m_dice3 = runSingle(num, n, "2", calCC, calGC,
                                                 calGD, calDice, onlySave)
