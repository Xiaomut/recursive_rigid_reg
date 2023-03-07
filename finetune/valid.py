import os
import sys
import torch
import numpy as np

sys.path.append("../")
sys.path.append("./")
from log import Log
from models.net import Net
from utils.base_util import readNiiImage, resampleNiiImg, loadJson, saveJson
from utils.image_util import cropImageByPoint, processOutPt, reviseMtxFromCrop, composeMatrixFromDegree


def imgToTensor(img):
    return torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)


def getImgs(num, base_dir):
    imgA = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"))
    gt_imgA = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

    return imgA, gt_imgA, imgB, gt_imgB


def loadModel(model_path):
    """ load the net """
    net = Net(8, 4)
    net_params = torch.load(model_path,
                            map_location=torch.device('cpu'))["net"]
    net.load_state_dict(net_params)
    net.eval()
    return net


def getCoord(num):
    shape = (481, 481, 481)
    # 更新一下坐标, 防止溢出
    coorA = r[f"img{num}"]["imgA"]
    coorA = processOutPt(coorA, shape, limit)
    coorB = r[f"img{num}"]["imgB"]
    coorB = processOutPt(coorB, shape, limit)

    return coorA, coorB


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

    imgA_tensor = imgToTensor(imgA_crop)
    imgB_tensor = imgToTensor(imgB_crop)

    pred_degree = net(imgA_tensor, imgB_tensor)
    fake_mtx = composeMatrixFromDegree(pred_degree.detach())
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


if __name__ == "__main__":

    data_path = "Y:/testdata/"
    r = loadJson("files/test_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]

    # need change before running codes
    model_path = "finetune/results/best_finetune.pth"
    resample_name = "exp2_finetune_now.nii.gz"  # _la2.5  _la2  _la1

    log = Log(filename="finetune/save_photo.log", mode="a").getlog()

    net = loadModel(model_path)
    num = 4
    _, _, imgB, gt_imgB = warp(num, net, True)