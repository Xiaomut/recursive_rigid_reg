import os
import numpy as np
import torch

from log import Log
from utils.image_util import invMatrix, decomposeMatrixDegree, composeMatrixFromDegree, reviseMtxFromCrop, cropImageByPoint, processOutPt
from utils.base_util import readNiiImage, saveNiiImage, loadJson, resampleNiiImg
from config import Config as conf
from models import recurnet_4img, recurnet_cbct


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


def validation(filenum, datadir="testdata", ifsave=False):
    base_dir = f"{conf.root_path}/{datadir}/img{filenum}"
    log.info(f"--------- filedir: [{datadir} -> img{filenum}] ---------")

    imgA, infos = readNiiImage(os.path.join(base_dir, "imgA.nii.gz"), True)
    gt_imgA = readNiiImage(os.path.join(base_dir, "gt_imgA.nii.gz"))
    imgB = readNiiImage(os.path.join(base_dir, "imgB.nii.gz"))
    gt_imgB = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"))

    limit = [200, -60, 128, 128, 128, 128]
    if "train" in datadir:
        r = loadJson(f"files/train_coordinate.json")
    else:
        r = loadJson(f"files/test_coordinate.json")

    # 更新一下坐标, 防止溢出
    coorA = r[f"img{filenum}"]["imgA"]
    coorA = processOutPt(coorA, imgA.shape, limit)
    coorB = r[f"img{filenum}"]["imgB"]
    coorB = processOutPt(coorB, imgB.shape, limit)

    # 截取原图像
    imgA_crop = cropImageByPoint(imgA, coorA, limit)
    imgB_crop = cropImageByPoint(imgB, coorB, limit)
    # saveNiiImage(imgA_crop, infos, os.path.join(base_dir, "imgA_crop.nii.gz"))
    # saveNiiImage(imgB_crop, infos, os.path.join(base_dir, "imgB_crop.nii.gz"))

    # 截取分割图像并保存
    imgA_gt_crop = cropImageByPoint(gt_imgA, coorA, limit)
    imgB_gt_crop = cropImageByPoint(gt_imgB, coorB, limit)
    saveNiiImage(imgA_gt_crop, infos,
                 os.path.join(base_dir, "gt_imgA_crop.nii.gz"))
    saveNiiImage(imgB_gt_crop, infos,
                 os.path.join(base_dir, "gt_imgB_crop.nii.gz"))

    # 得到下颌骨剪切后的图像并保存
    imgA_input = imgA_crop * imgA_gt_crop
    imgB_input = imgB_crop * imgB_gt_crop
    saveNiiImage(imgA_input, infos, os.path.join(base_dir, "imgA_crop.nii.gz"))
    saveNiiImage(imgB_input, infos, os.path.join(base_dir, "imgB_crop.nii.gz"))

    imgA_tensor = torch.FloatTensor(imgA_crop).unsqueeze(0).unsqueeze(0)
    imgB_tensor = torch.FloatTensor(imgB_crop).unsqueeze(0).unsqueeze(0)
    imgB_ori_tensor = torch.FloatTensor(imgB).unsqueeze(0).unsqueeze(0)

    imgA_gt_tensor = torch.FloatTensor(imgA_gt_crop).unsqueeze(0).unsqueeze(0)
    imgB_gt_tensor = torch.FloatTensor(imgB_gt_crop).unsqueeze(0).unsqueeze(0)

    imgA_tensor, imgB_tensor = imgA_tensor.to(device), imgB_tensor.to(device)
    imgA_gt_tensor = imgA_gt_tensor.to(device)
    imgB_gt_tensor = imgB_gt_tensor.to(device)

    stem_results, thetas, stem_results_gt = net(imgA_tensor, imgB_tensor,
                                                imgA_gt_tensor, imgB_gt_tensor)
    log.info([theta.detach() for theta in thetas])
    save_name = os.path.join(base_dir, resample_name)
    if ifsave:
        thetas_ori = getOriThetas(thetas, coorA, coorB, imgA.shape,
                                  imgB_crop.shape)
        warped = imgB_ori_tensor
        for i, (stem_result, theta_o, stem_result_gt) in enumerate(
                zip(stem_results, thetas_ori, stem_results_gt)):
            # 需要将其相乘进行存储
            warped_gt = (stem_result *
                         stem_result_gt).squeeze(0).squeeze(0).cpu().detach()
            # 保存被一个阶段的图像
            # saveNiiImage(warped_gt.numpy(), infos, save_name.replace(".nii.gz", f"_crop_{nums}{i+1}.nii.gz"))
            # warped = resampleNiiImg(theta_o, warped, infos, save_name.replace(".nii.gz", f"_{nums}{i+1}.nii.gz"))
            # 只保存最后的图像
            warped = resampleNiiImg(theta_o, warped, infos)
        warped = warped.type(torch.ShortTensor).squeeze().squeeze().numpy()
        saveNiiImage(warped_gt.numpy(), infos,
                     save_name.replace(".nii.gz", "_crop_final.nii.gz"))
        saveNiiImage(warped, infos,
                     save_name.replace(".nii.gz", "_final.nii.gz"))


if __name__ == "__main__":

    model_path = r"recurse\cas3\ori_corr_0832\best_recurse.pth"
    state_dict = torch.load(model_path, map_location="cpu")

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = recurnet_4img.RecursiveCascadeNetwork(conf.n_cascades,
    #                                               conf.channel, device,
    #                                               state_dict, True)
    net = recurnet_cbct.RecursiveCascadeNetwork(conf.n_cascades, 8, 32, device,
                                                True, True, state_dict, True)

    log = Log(filename=f"{conf.save_name}/valid/validation.log",
              mode="a").getlog()
    log.info(f"------- {model_path} -------")
    log.info(state_dict["epoch"])
    resample_name = f"warped_{conf.save_name}.nii.gz"

    for filenum in range(26, 28):
        validation(filenum, datadir="testdata", ifsave=True)

    # validation(83, datadir="traindata2", ifsave=False)
