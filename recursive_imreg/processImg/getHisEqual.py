import os
import sys
import numpy as np

sys.path.append('../')
sys.path.append('./')
from utils import base_util, image_util, histmatching


def getBaseImg():
    baseimg = "Y:/traindata2/img1/imgA.nii.gz"
    coord = [248, 271, 338]
    limit = [220, -80, 128, 128, 128, 128]

    imgA, infos = base_util.readNiiImage(baseimg, True)
    coord_A = image_util.processOutPt(coord, imgA.shape, limit)
    imgA_crop = image_util.cropImageByPoint(imgA, coord_A, limit)
    # save
    os.makedirs("Y:/traindata2/baseimg", exist_ok=True)
    base_util.saveNiiImage(imgA_crop, infos,
                           "Y:/traindata2/baseimg/base.nii.gz")


def getBaseInfo():
    file = "Y:/traindata2/baseimg/base.nii.gz"
    template, infos = base_util.readNiiImage(file, True)
    nt_data_array = template.ravel()
    t_values, t_counts = np.unique(nt_data_array, return_counts=True)
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    return t_quantiles, t_values, infos


def getCropImg(imgrec, imgnum, r, limit):
    basedir = "Y:/traindata2"
    filename = os.path.join(basedir, imgnum, f"img{imgrec}.nii.gz")

    img = base_util.readNiiImage(filename)

    coord = r[imgnum][f"img{imgrec}"]
    new_coord = image_util.processOutPt(coord, img.shape, limit)

    img_crop = image_util.cropImageByPoint(img, new_coord, limit)
    return img_crop, filename


def getAllHisImgs():
    r = base_util.loadJson("files/train_coordinate.json")
    limit = [220, -80, 128, 128, 128, 128]

    t_quantiles, t_values, infos = getBaseInfo()

    errors = []
    for i in range(2, 124):
        imgA, fileA = getCropImg("A", f"img{i}", r, limit)
        imgB, fileB = getCropImg("B", f"img{i}", r, limit)
        saveA = fileA.replace("imgA.nii.gz", "imgA_crop_his.nii.gz")
        saveB = fileB.replace("imgB.nii.gz", "imgB_crop_his.nii.gz")
        try:
            histmatching.histForSave(t_quantiles, t_values, imgA, saveA, infos)
        except Exception as e:
            errors.append(fileA)
            print(f"The error is {e}")
        try:
            histmatching.histForSave(t_quantiles, t_values, imgB, saveB, infos)
        except Exception as e:
            errors.append(fileA)
            print(f"The error is {e}")
        print(f"---------- img{i} has done ----------")
    print(errors)


if __name__ == "__main__":
    # 得到base图像
    # getBaseImg()
    # 处理所有图像
    # getAllHisImgs()
