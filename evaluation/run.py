import re
import numpy as np
from glob import glob

from base_util import loadJson, saveJson
from log import Log


def loadlog(file):
    with open(file, 'r', encoding='utf-8') as f:
        r = f.read()
    return r


def extractCC(text, filetype='2'):
    nums = re.findall(f"Type {filetype}.*?\[cc: (-?\d+\.\d+)\]", text)
    return [float(num) for num in nums]


def extractGC(text, filetype='2'):
    nums = re.findall(f"Type {filetype}.*?\[gc: (-?\d+\.\d+)\]", text)
    return [float(num) for num in nums]


def extractGD(text, filetype='2'):
    nums = re.findall(f"Type {filetype}.*?\[gd: (-?\d+\.\d+)\]", text)
    return [float(num) for num in nums]


def extractDice(text, filetype='2'):
    nums = re.findall(f"Type {filetype}.*?\[dice: (-?\d+\.\d+)\]", text)
    return [float(num) for num in nums]


def calMeanStd(data):
    data = np.asarray(data)
    return np.mean(data), np.std(data)


def getRes(file, regtype='0', index="cc"):
    text = loadlog(file)
    if index == "cc":
        return calMeanStd(extractCC(text, regtype))
    elif index == "gc":
        return calMeanStd(extractGC(text, regtype))
    elif index == "gd":
        return calMeanStd(extractGD(text, regtype))
    elif index == "dice":
        return calMeanStd(extractDice(text, regtype))
    else:
        raise NotImplementedError("Not NotImplemented!")


def getMetrics(file=r"exp2\log\exp2_la1.log", regType='2'):
    cc_mean, cc_std = getRes(file, regType, "cc")
    gc_mean, gc_std = getRes(file, regType, "gc")
    gd_mean, gd_std = getRes(file, regType, "gd")
    try:
        dice_mean, dice_std = getRes(file, regType, "dice")
    except:
        dice_mean, dice_std = "nan", "nan"
    print(f"-------------- file: [{file}] --------------")
    print(
        f"regType {regType} -> [cc: {cc_mean:.4f}, {cc_std:.4f}] -> [gc: {gc_mean:.4f}, {gc_std:.4f}] -> [gd: {gd_mean:.4f}, {gd_std:.4f}] -> [dice: {dice_mean:.4f}, {dice_std:.4f}]"
    )


def getBaseInfo(file=r"exp2\log\exp2_.log"):
    """ gets `elastix` and `sift` results from exp2_.log """
    for i in range(2):
        getMetrics(file, str(i))


def getExpAll(exp_num, rules='*', regType='2'):
    """
    @param exp_num: int. 1, 2, 3 decide which experience
    @param rules: string. define the rules to filt the files. default '*', means all files
    @param regType: string. `0` is elastix, `1` is sift, `2` is exp. default `2`.
    """
    files = sorted(glob(f"exp{exp_num}/log/{rules}.log"))
    for file in files:
        try:
            getMetrics(file, regType)
        except:
            pass


def contract(num=1):
    if num == 1:
        getBaseInfo("exp1/log/exp1.log")
    elif num == 2 or num == 3:
        getBaseInfo("exp2/log/exp2_.log")
    # elif num == 3:
    #     getBaseInfo("exp3/exp3_base.log")
    print('$' * 20)
    getExpAll(num, '*', '2')


if __name__ == "__main__":

    # contract diff scale fcc
    # getExpAll(3, 'cas1_*', '2')
    
    getMetrics(3, 'cas1_*', '2')
    # contract(3)