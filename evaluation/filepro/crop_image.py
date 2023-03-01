import os
import numpy as np
from base_util import readNiiImage, saveNiiImage, loadJson


def cropImageByCenter(array, size=256):
    """ Crop image by Center. give the output `size` """
    assert len(array.shape) == 3, "array must be 3D image"
    ori_D, ori_H, ori_W = array.shape

    array = array[(ori_D - size) // 2:(ori_D - size) // 2 + size,
                  (ori_H - size) // 2:(ori_H - size) // 2 + size,
                  (ori_W - size) // 2:(ori_W - size) // 2 + size]
    return array


def cropImageByPoint(array, pt, limit):
    """ 
    Crop image by point. set image as (140, 256, 256)
    """
    p1, p2, p3, p4, p5, p6 = pt[2] - limit[0], pt[2] + limit[1], pt[1] - limit[
        2], pt[1] + limit[3], pt[0] - limit[4], pt[0] + limit[5]
    array = array[p1:p2, p3:p4, p5:p6]
    return array


def loadImg(imgnum, imgrec):
    """ load image by `num` and `A` or `B` """
    base_path = os.path.join(data_path, f"img{imgnum}")
    basedir = os.path.join(base_path, f"img{imgnum}")

    img_file = os.path.join(basedir, f"img{imgrec}.nii.gz")
    img_gt_file = os.path.join(basedir, f"gt_img{imgrec}.nii.gz")

    img, infos = readNiiImage(img_file, True)
    img_gt = readNiiImage(img_gt_file)

    return img, img_gt, img_file, img_gt_file, infos


def changeName(img_file, img_gt_file, repname):
    """ replace name """
    img_crop_name = img_file.replace("imgA", f"imgA_{repname}")
    img_crop_name = img_crop_name.replace("imgB", f"imgB_{repname}")
    gt_crop_name = img_gt_file.replace("gt_imgA", f"gt_imgA_{repname}")
    gt_crop_name = gt_crop_name.replace("gt_imgB", f"gt_imgB_{repname}")
    return img_crop_name, gt_crop_name


def processOutPt(pt, img_shape, limit):
    """ limit the output boundary in 481 """
    assert len(img_shape) == 3, "array must be 3D image"
    ori_D, ori_H, ori_W = img_shape

    limits = [
        pt[2] - limit[0], pt[2] + limit[1], pt[1] - limit[2], pt[1] + limit[3],
        pt[0] - limit[4], pt[0] + limit[5]
    ]
    if max(limits) < ori_D and min(limits) >= 0:
        return pt
    else:
        if limits[0] < 0:
            pt[2] = limit[0] + 1
        if limits[1] > ori_D:
            pt[2] = ori_D - limit[1] - 1
        if limits[2] < 0:
            pt[1] = limit[2] + 1
        if limits[3] > ori_H:
            pt[1] = ori_H - limit[3] - 1
        if limits[4] < 0:
            pt[0] = limit[4] + 1
        if limits[5] > ori_W:
            pt[0] = ori_W - limit[5] - 1
        return pt


def saveCtCropImg(imgnum, imgrec, ifdot=False, repname="ct_crop"):
    """
    @param imgnum: int. img file num, such as 1, 2,..., 44
    @param imgrec: `A`, `B`. fixed image or moving image, `A` is reference image `B` is moving image.
    @param ifdot: bool. True means [img * segment]. default False
    """
    # read image
    img, img_gt, img_file, img_gt_file, infos = loadImg(imgnum, imgrec)
    img_crop = cropImageByCenter(img, size=256)
    img_gt_crop = cropImageByCenter(img_gt, size=256)

    if ifdot:
        img_save = img_crop * img_gt_crop
    else:
        img_save = img_crop

    # change name
    img_crop_name, gt_crop_name = changeName(img_file, img_gt_file, repname)

    saveNiiImage(img_save, infos, img_crop_name)
    saveNiiImage(img_gt_crop, infos, gt_crop_name)


def savePtCropImg(imgnum,
                  imgrec,
                  use_valid=False,
                  ifdot=False,
                  repname="pt_crop"):
    """
    @param imgnum: int. img file num, such as 1, 2,..., 44
    @param imgrec: `A`, `B`. fixed image or moving image, `A` is reference image `B` is moving image.
    @param use_valid: bool. True means compare two images based on imgA. default False
    @param ifdot: bool. True means [img * segment]. default False
    """
    # process coordinate
    r = loadJson("files/test_coordinate.json")
    limit = [200, -60, 128, 128, 128, 128]

    if use_valid:
        coord = r[f"img{imgnum}"][f"imgA"]
    else:
        coord = r[f"img{imgnum}"][f"img{imgrec}"]

    # read image
    img, img_gt, img_file, img_gt_file, infos = loadImg(imgnum, imgrec)

    new_coord = processOutPt(coord, img.shape, limit)
    img_crop = cropImageByPoint(img, new_coord, limit)
    img_gt_crop = cropImageByPoint(img_gt, new_coord, limit)

    if ifdot:
        img_save = img_crop * img_gt_crop
    else:
        img_save = img_crop

    # change name
    img_crop_name, gt_crop_name = changeName(img_file, img_gt_file, repname)

    saveNiiImage(img_save, infos, img_crop_name)
    saveNiiImage(img_gt_crop, infos, gt_crop_name)


if __name__ == "__main__":
    data_path = "Y:/testdata/"
    num = 1

    savePtCropImg(num, "A", False, repname="pt_crop")