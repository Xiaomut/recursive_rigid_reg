import os
import numpy as np
import torch
import sys

sys.path.append("../")
sys.path.append("./")
from utils.image_util import invMatrix, decomposeMatrixDegree, composeMatrixFromDegree, reviseMtxFromCrop, cropImageByPoint, processOutPt
from utils.base_util import readNiiImage, saveNiiImage, loadJson, resampleNiiImg


def transSift(num):
    base_dir = f"Y:/testdata/img{num}"
    gt_img, infos = readNiiImage(os.path.join(base_dir, "gt_imgB.nii.gz"),
                                 True)
    label = np.loadtxt(os.path.join(base_dir, "label.txt"))
    label = torch.from_numpy(label).view(1, 3, 4)

    imgA_tensor = torch.FloatTensor(gt_img).unsqueeze(0).unsqueeze(0)
    resampleNiiImg(label, imgA_tensor, infos,
                   os.path.join(base_dir, "gt_sift.nii.gz"), "nearest")
    print(f"-------- {num} done! --------")


if __name__ == "__main__":
    for i in range(2, 45):
        transSift(i)