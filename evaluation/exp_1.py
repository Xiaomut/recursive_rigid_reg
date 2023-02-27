import os
import numpy as np
import torch
import torch.nn.functional as F

from log import Log
from utils.image_util import invMatrix, decomposeMatrixDegree, composeMatrixFromDegree, reviseMtxFromCrop, cropImageByPoint, processOutPt
from utils.base_util import readNiiImage, saveNiiImage, loadJson, resampleNiiImg


def validation(filenum, net_mode="test", ifsave=False):
    base_dir = f"{conf.root_path}/{net_mode}data/img{filenum}"
    log.info(f"--------- filedir: [{net_mode} -> img{filenum}] ---------")

    imgA, infos = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"), True)
    gt_imgA = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))
    # label_mtx = np.loadtxt(os.path.join(base_dir, "label.txt"))

    r = loadJson(f"files/train_coordinate.json")
    limit = [256, -60, 200, 128, 128, 128]

    # 更新一下坐标, 防止溢出
    coorA = r[f"img{filenum}"]["imgA"]
    coorA = processOutPt(coorA, imgA.shape, limit)
    coorB = r[f"img{filenum}"]["imgB"]
    coorB = processOutPt(coorB, imgB.shape, limit)

    # 截取原图像
    imgA_crop = cropImageByPoint(imgA_input, coorA, limit)
    imgB_crop = cropImageByPoint(imgB_input, coorB, limit)

    # 截取分割图像
    imgA_gt_crop = cropImageByPoint(imgA_gt, coorA, limit)
    imgB_gt_crop = cropImageByPoint(imgB_gt, coorB, limit)

    # 得到下颌骨剪切后的图像
    imgA_input = imgA_crop * imgA_gt_crop
    imgB_input = imgB_crop * imgB_gt_crop
    saveNiiImage(imgA_input, infos, os.path.join(base_dir, "imgA_crop.nii.gz"))
    saveNiiImage(imgB_input, infos, os.path.join(base_dir, "imgB_crop.nii.gz"))

    imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0).unsqueeze(0)
    imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0).unsqueeze(0)

    imgA_gt_tensor = torch.FloatTensor(imgA_gt_crop).unsqueeze(0).unsqueeze(0)
    imgB_gt_tensor = torch.FloatTensor(imgB_gt_crop).unsqueeze(0).unsqueeze(0)

    imgA_tensor, imgB_tensor = imgA_tensor.to(device), imgB_tensor.to(device)
    imgA_gt_tensor, imgB_gt_tensor = imgA_gt_tensor.to(
        device), imgB_gt_tensor.to(device)

    stem_results, thetas = net(imgA_tensor, imgB_tensor)
    log.info(np.asarray([theta.cpu().detach().numpy() for theta in thetas]))
    save_name = os.path.join(base_dir, resample_name)
    if ifsave:
        for i, stem_result in enumerate(stem_results):
            resample = stem_result.squeeze(0).squeeze(0).cpu().detach()
            saveNiiImage(resample.numpy(), infos,
                         save_name.replace(".nii.gz", f"crop_{i+1}.nii.gz"))


if __name__ == "__main__":

    pass