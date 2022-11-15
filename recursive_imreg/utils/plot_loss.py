import re
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
        print(train_sum.index(min_num), min_num)
        plt.plot(range(len(train_sum)), train_sum, label="train")
    elif mode == "1":
        min_num = min(test_sum)
        print(test_sum.index(min_num), min_num)
        plt.plot(range(len(test_sum)), test_sum, label="test")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    file = r"recurse\cas3\small\train.log"
    plotLog(file, '1')
    # plotLabel()