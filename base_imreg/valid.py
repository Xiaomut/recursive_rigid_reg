import os
import numpy as np
import torch
import torch.nn.functional as F

from log import Log
from utils.base_util import getImageDirs, loadJson, readNiiImage, saveNiiImage, resampleNiiImg
from utils.image_util import cropImageByCenter, invMatrix, decomposeMatrixDegree, imgnorm, reviseMtxToCrop, composeMatrixFromDegree, reviseMtxFromCrop
from config import Config as conf
from models import model


def validation(filenum, net_mode="test", ifsave=False):
    base_dir = f"{conf.root_path}/{net_mode}data/img{filenum}"
    log.info(f"--------- filedir: [{net_mode} -> img{filenum}] ---------")

    imgA, infos = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"), True)
    # gt_imgA = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))
    label_mtx = np.loadtxt(os.path.join(base_dir, "label.txt"))

    size = 256
    shape = (size, size, size)

    label_mtx_reivse = reviseMtxToCrop(imgA.shape, shape, label_mtx)
    label_degree = decomposeMatrixDegree(label_mtx_reivse, change=conf.change)

    # 截取原图像
    imgA_crop = cropImageByCenter(imgA, size=size)
    imgB_crop = cropImageByCenter(imgB, size=size)

    # imgA_crop = cropImageByCenter(imgA, size=size)
    imgB_gt_crop = cropImageByCenter(gt_imgB, size=size)

    imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0).unsqueeze(0)
    imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0).unsqueeze(0)

    # imgA_gt_tensor = torch.FloatTensor(imgA_gt_crop).unsqueeze(0).unsqueeze(0)

    # 转换成张量
    imgA_tensor, imgB_tensor = imgA_tensor.to(device), imgB_tensor.to(device)
    # imgA_gt_tensor, imgB_gt_tensor = imgA_gt_tensor.to(device), imgB_gt_tensor.to(device)
    label_tensor = torch.FloatTensor(label_degree).to(device)

    # pred
    pred_degree = net(imgA_tensor, imgB_tensor)

    # 计算损失
    criterion = torch.nn.MSELoss().to(device)
    loss = criterion(label_tensor.view(pred_degree.size()), pred_degree).item()
    log.info(f"MSEloss: {loss:.4f}")

    # 计算转换后的变换矩阵
    fake_mtx = composeMatrixFromDegree(pred_degree.detach().numpy(),
                                       change=conf.change)
    log.info(f'Pred: {fake_mtx.cpu().detach().numpy()}')

    real_mtx = composeMatrixFromDegree(label_degree, change=conf.change)
    log.info(f'Real: {real_mtx}')

    if ifsave:
        ori_save_name = os.path.join(base_dir, resample_name)
        seg_name = ori_save_name.replace("_now", "_seg")
        pred_mtx_revise = reviseMtxFromCrop(imgA.shape, shape, fake_mtx)
        img_o_tensor = torch.FloatTensor(imgB).unsqueeze(0).unsqueeze(0)
        imgB_gt_tensor = torch.FloatTensor(imgB_gt_crop).unsqueeze(
            0).unsqueeze(0)
        resampleNiiImg(pred_mtx_revise, img_o_tensor, infos, ori_save_name)
        resampleNiiImg(pred_mtx_revise, imgB_gt_tensor, infos, seg_name)


if __name__ == "__main__":

    model_path = r"result\pth1632\result_B2A.pth"
    # 一些基本设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = model.Net(conf.midch, conf.growthrate,
                    [1, 2, 4, 8, 16, 32]).to(device)
    try:
        net.load_state_dict(torch.load(model_path))
    except:
        net.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()

    log = Log(filename="log/validation.log", mode="w").getlog()

    # net_mode 决定训练集还是测试集, ifsave决定是否保存
    # for filenum in range(82, 86):
    #     validation(filenum, net_mode="train", ifsave=False)

    resample_name = "exp1_now.nii.gz"
    filenum = 1
    validation(filenum, net_mode="test", ifsave=True)
