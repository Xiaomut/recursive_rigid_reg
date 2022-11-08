import os
import torch
import torch.nn.functional as F

from log import Log
from utils.base_util import readNiiImage, saveNiiImage, loadJson, saveJson
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


def getJson():
    jsonfile = "result/train_coordinate.json"
    base_dir = "Y:/traindata2"
    images_dir = loadJson(jsonfile)
    for k, v in images_dir.items():
        imageA = os.path.join(base_dir, k, "imgA.nii.gz")
        pred_coord_A, _ = pred(imageA)
        v["imgA"] = pred_coord_A.numpy()[::-1]
        imageB = os.path.join(base_dir, k, "imgB.nii.gz")
        pred_coord_B, _ = pred(imageB)
        v["imgB"] = pred_coord_B.numpy()[::-1]
    saveJson(images_dir, jsonfile)


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
    getJson()