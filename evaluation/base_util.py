import os
import json
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def getImageDirs(root_dir):
    """获取所有图像路径"""
    image_dirs = os.listdir(root_dir)
    image_dirs_cat = sorted([os.path.join(root_dir, i) for i in image_dirs])
    return image_dirs_cat


def loadJson(file):
    with open(file, 'r') as f:  # , encoding="utf-8"
        r = json.load(f)
    return r


def saveJson(adict, save_file):
    """存json格式的文件"""
    with open(save_file, 'w') as f:
        json.dump(adict, f)


def saveJsonAdd(adict, save_file):
    """追加模式存取json, 方便读取"""
    if os.path.isfile(save_file) and os.path.getsize(save_file) > 0:
        r = loadJson(save_file)
        with open(save_file, 'w') as f:
            # 添加内容
            r = dict(r, **adict)
            json.dump(r, f)
    else:
        with open(save_file, 'w') as f:
            json.dump(adict, f)


def readNiiImage(file, otherinfo=False):
    """读取图像, 如果需要其他信息, 则为True"""
    img = sitk.ReadImage(file)
    img_array = sitk.GetArrayFromImage(img)
    if otherinfo:
        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()
        infos = {"o": origin, "s": spacing, "d": direction}
        return img_array, infos
    return img_array


def saveNiiImage(array, infos, filename):
    """保存为nii图像"""
    img = sitk.GetImageFromArray(array.astype(np.int16))
    img.SetOrigin(infos['o'])
    img.SetSpacing(infos['s'])
    img.SetDirection(infos['d'])
    sitk.WriteImage(img, fileName=filename)


def resampleNiiImg(mtx, img, infos=None, save_file=None, mode="bilinear"):
    """对输入图像进行重采样, mode 可选 ['bilinear', 'nearest']"""
    grid = F.affine_grid(mtx, img.size(), align_corners=False).float()
    resample = F.grid_sample(img, grid, mode=mode, align_corners=False)
    if save_file is not None:
        resample_save = resample.type(torch.ShortTensor).squeeze().squeeze()
        saveNiiImage(resample_save.numpy(), infos, save_file)
    return resample