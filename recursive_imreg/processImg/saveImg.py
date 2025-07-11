import os
import sys

sys.path.append('../')
sys.path.append('./')
from utils import base_util, image_util, histmatching


def saveCropImg(imgnum,
                imgrec,
                base="Y:/traindata2",
                jsonfile="files/train_coordinate.json"):
    r = base_util.loadJson(jsonfile)
    # limit = [220, -80, 128, 128, 128, 128]
    limit = [200, -60, 128, 128, 128, 128]

    basedir = os.path.join(base, f"img{imgnum}")

    img_file = os.path.join(basedir, f"img{imgrec}.nii.gz")
    img_gt_file = os.path.join(basedir, f"gt_img{imgrec}.nii.gz")

    coord = r[f"img{imgnum}"][f"img{imgrec}"]

    img, infos = base_util.readNiiImage(img_file, True)
    img_gt = base_util.readNiiImage(img_gt_file)

    new_coord = image_util.processOutPt(coord, img.shape, limit)
    img_crop = image_util.cropImageByPoint(img, new_coord, limit)
    img_gt_crop = image_util.cropImageByPoint(img_gt, new_coord, limit)

    img_save = img_crop * img_gt_crop

    img_crop_name = img_file.replace("imgA",
                                     "imgA_crop").replace("imgB", "imgB_crop")
    gt_crop_name = img_gt_file.replace("gt_imgA", "gt_imgA_crop").replace(
        "gt_imgB", "gt_imgB_crop")
    # base_util.saveNiiImage(img_save, infos, img_crop_name)
    base_util.saveNiiImage(img_gt_crop, infos, gt_crop_name)


if __name__ == "__main__":
    # num = 25
    base = "Y:/traindata2"  # traindata2  testdata
    jsonfile = "files/train_coordinate.json"  # test_coordinate
    for num in range(4, 124):
        saveCropImg(num, "A", base, jsonfile)
        saveCropImg(num, "B", base, jsonfile)