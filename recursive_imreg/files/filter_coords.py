import re
import sys
import numpy as np

sys.path.append("../")
sys.path.append("./")
from utils.base_util import readNiiImage, loadJson


def processOutPt(pt, img_shape, limit):
    """ 限制输出点的范围 """
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
        return False


def filterPt(file="files/train_coordinate.json"):
    r = loadJson(file)
    # r = loadJson("files/test_coordinate.json")
    img_shape = (481, 481, 481)
    errors = []
    # limit = [256, -60, 200, 128, 128, 128]
    limit = [200, -60, 196, 128, 128, 128]
    for k, v in r.items():
        # print(f"----------- {k} -----------")
        vA = v["imgA"]
        vA = processOutPt(vA, img_shape, limit)
        vB = v["imgB"]
        vB = processOutPt(vB, img_shape, limit)
        if not vA or not vB:
            errors.extend(map(int, re.findall("\d+", k)))
    print("errors: ", errors)


if __name__ == "__main__":
    # pass
    filterPt("files/train_coordinate.json")