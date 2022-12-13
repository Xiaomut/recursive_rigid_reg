import re
import numpy as np
import matplotlib.pyplot as plt


def getMinNum(nums):
    min_num = min(nums)
    x = nums.index(min_num)
    y = np.round(min_num, 3)
    return x, y


def getInfos(logfile):
    with open(logfile, 'r') as f:
        r = f.read()

    train_sum = re.findall("\[train Loss: (\d+\.\d+)\]", r)
    valid_sum = re.findall("\[valid Loss: (\d+\.\d+)\]", r)
    test_sum = re.findall("\[test Loss: (\d+\.\d+)\]", r)

    train_sum = [float(i) for i in train_sum]
    valid_sum = [float(i) for i in valid_sum]
    test_sum = [float(i) for i in test_sum]

    train_x, train_y = getMinNum(train_sum)
    valid_x, valid_y = getMinNum(valid_sum)
    test_x, test_y = getMinNum(test_sum)
    return {
        "train": train_sum,
        "valid": valid_sum,
        "test": test_sum,
        "train_min": (train_x, train_y),
        "valid_min": (valid_x, valid_y),
        "test_min": (test_x, test_y)
    }


def plotLog(files):
    """ 由log文件生成loss图像 """

    figs, ax = plt.subplots(1, 3)

    for idx, (file, label) in enumerate(
            zip(files, ["origin", "fea_corr", "histq_corr"])):
        infos = getInfos(file)
        ax[0].plot(range(len(infos["train"])), infos["train"], label=label)
        ax[0].set_title("Train Loss")

        ax[1].plot(range(len(infos["valid"])), infos["valid"], label=label)
        ax[1].set_title("Valid Loss")

        ax[2].plot(range(len(infos["test"])), infos["test"], label=label)
        ax[2].set_title("Test Loss")

    #     plt.scatter(x, y, marker='*', c='r')
    #     # 第一个参数为标记文本，第二个参数为标记对象的坐标，第三个参数为标记位置
    #     plt.annotate(f'({x}, {y})', xy=(x, y), xytext=(x - 10, y + 1))
    #     plt.plot(range(len(test_sum)), test_sum, label="test")
    for x in ax:
        x.legend()
        x.grid()
    plt.show()


if __name__ == "__main__":
    files = [
        r"recurse\cas3\ori_8\train.log",
        r"recurse\cas3\ori_corr_8\train.log",
        r"recurse\cas3\fullhis_corr_8\train.log",
    ]
    # for file in files:
    #     plotLog(file, '1')
    plotLog(files)
