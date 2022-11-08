import os, re
import numpy as np
import torch
import torch.nn.functional as F

from log import Log
from utils.base_util import readNiiImage, saveNiiImage, loadJson
from utils import dataloads, loss
from model.coordnet import CoordRegressionNetwork


def validation(imgnum, net_mode="train", ab="a"):

    base_dir = f"/home/wangs/{net_mode}data" if os.path.exists(
        "/home") else f"Y:/{net_mode}data"
    filename = os.path.join(base_dir, f"img{imgnum}",
                            f"img{ab.upper()}.nii.gz")
    img = readNiiImage(filename)

    image_size = (size, size, size)
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

    model_path = "landmark/result/best_coords.pth"
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
    log = Log(filename="landmark/log/validation.log", mode="a").getlog()

    size = 128
    validation(imgnum=10, net_mode="test", ab="b")