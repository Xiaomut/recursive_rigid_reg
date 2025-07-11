import os, re
import numpy as np
import torch
import torch.nn.functional as F

from log import Log
from utils.base_util import readNiiImage, saveNiiImage, loadJson
from utils import dataloads, loss, image_util
from model.coordnet import CoordRegressionNetwork


def pred(imgfile, size=128):
    img = readNiiImage(imgfile)
    image_size = (size, size, size)

    img_new = image_util.resizeImg(img, image_size)
    img_tensor = torch.FloatTensor(img_new).unsqueeze(0).unsqueeze(0).to(
        device)

    # 网络得到的坐标点
    pred_ori, heatmaps = net(img_tensor)

    # 还原到原图像大小
    pred_coord = ((pred_ori.cpu().detach().squeeze(0).squeeze(0) + 1) *
                  torch.Tensor(image_size) - 1) / 2
    pred_coord = (pred_coord * 481 / size).type(torch.int16)
    print(f"pred_coord: {pred_coord.numpy()[::-1]}")
    return pred_coord, heatmaps


def validation(imgnum):

    base_dir = ""
    filename = ""
    img = readNiiImage(filename)

    r = loadJson(f"part5/jsonfile/{net_mode}_coordinate.json")

    base1 = os.path.basename(filename)[:4]
    base2 = re.findall("img\d+", filename)[0]
    coord_ori = r[base2][base1]

    img_new, coord_new = dataloads.resizeImg(img, coord_ori)
    # coord = np.asarray(coord_ori[::-1]) - 112

    img_tensor = torch.FloatTensor(img_new).unsqueeze(0).unsqueeze(0).to(
        device)
    coord_tensor = torch.Tensor(coord_new)
    coord_tensor = (coord_tensor * 2 + 1) / torch.Tensor(image_size) - 1

    pred, heatmaps = net(img_tensor)
    print(f"pred: {pred.cpu().detach().numpy()}")
    print(f"label: {coord_tensor}")

    pred_coord = ((pred.cpu().detach().squeeze(0).squeeze(0) + 1) *
                  torch.Tensor(image_size) - 1) / 2
    pred_coord = pred_coord * 481 / size
    pred_coord = pred_coord.type(torch.int16)
    print(f"pred_coord: {pred_coord.numpy()[::-1]}")
    print(f"ori_coord: {coord_ori}")


if __name__ == "__main__":

    model_path = "result/best_coords.pth"
    # 一些基本设置
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = CoordRegressionNetwork(n_locations=1,
                                 net="unet",
                                 conv_depths=(8, 16, 32, 64),
                                 midch=8).to(device)
    try:
        net.load_state_dict(torch.load(model_path)["net"])
    except:
        net.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))["net"])
    net.eval()
    log = Log(filename="log/validation.log", mode="a").getlog()

    size = 128
    # validation(imgnum=10, net_mode="test", ab="b")

    # file = "Y:/testdata/img1/imgA.nii.gz"
    file = "Y:/traindata2/img1/imgB.nii.gz"
    pred_coor, heat = pred(file)