import sys

sys.path.append("../")
sys.path.append("./")
from utils import base_util, image_util


def initJson():
    jsonfile = "files/train_coordinate.json"
    dic = {}
    for i in range(1, 124):
        dic[f"img{i}"] = {"imgA": [], "imgB": []}
    base_util.saveJson(dic, jsonfile)


if __name__ == "__main__":
    initJson()