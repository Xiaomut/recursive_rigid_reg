import re
import numpy as np
import matplotlib.pyplot as plt


def plotLog(logfile, mode="0"):
    """ 由log文件生成loss图像 """
    with open(logfile, 'r') as f:
        r = f.read()

    train_sum = re.findall("\[train Loss: (\d+\.\d+)\]", r)
    test_sum = re.findall("\[test Loss: (\d+\.\d+)\]", r)

    train_sum = [float(i) for i in train_sum]
    test_sum = [float(i) for i in test_sum]

    if mode == "0":
        min_num = min(train_sum)
        x = train_sum.index(min_num)
        y = np.round(min_num, 3)
        print(x, y)
        plt.title("Train Loss")
        plt.scatter(x, y, marker='*', c='r')
        plt.annotate(f'({x}, {y})', xy=(x, y), xytext=(x - 10, y + 3))
        plt.plot(range(len(train_sum)), train_sum, label="train")
    elif mode == "1":
        min_num = min(test_sum)
        x = test_sum.index(min_num)
        y = np.round(min_num, 3)
        print(x, y)
        plt.title("Valid Loss")
        plt.scatter(x, y, marker='*', c='r')
        # 第一个参数为标记文本，第二个参数为标记对象的坐标，第三个参数为标记位置
        plt.annotate(f'({x}, {y})', xy=(x, y), xytext=(x - 10, y + 1))
        plt.plot(range(len(test_sum)), test_sum, label="test")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    file = r"recurse\cas3\small_his\train.log"
    plotLog(file, '1')
    # plotLabel()