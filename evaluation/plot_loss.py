import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import seaborn as sns
import run
from run import loadlog

plt.rc('font', family='Times New Roman')


def _5_3_diff(filetype=0):
    values = np.asarray([[0.8165, 12.1183, 0.8865], [0.845, 11.7894, 0.9052],
                         [0.8428, 11.591, 0.9025], [0.8565, 11.5008, 0.9085],
                         [0.8536, 11.4475, 0.9067], [0.861, 11.2416, 0.913],
                         [0.8306, 11.5496, 0.89], [0.8407, 11.6818, 0.8974]])

    x_ticks = [
        'n=1', 'Corr,n=1', 'n=2', 'Corr,n=2', 'n=3', 'Corr,n=3', 'n=4',
        'Corr,n=4'
    ]

    if filetype == 0:
        plt.plot(np.arange(8), values[:, 0], label="NCC", color="#5555FF")
        plt.plot(np.arange(8), values[:, 2], label="Dice", color="#FF7744")
        plt.bar(np.arange(8) + 0.1, values[:, 0], width=0.05, color="#5555FF")
        plt.bar(np.arange(8) + 0.2, values[:, 2], width=0.05, color="#FF7744")
        plt.scatter(5, values[5, 0], color='#D9006C', s=50, marker='*')
        plt.scatter(5, values[5, 2], color='#D9006C', s=50, marker='*')
        plt.ylim((0.8, 0.95))
        plt.title("Results comparison about NCC and Dice")
    elif filetype == 1:
        plt.plot(np.arange(8), values[:, 1], label="GD", color="#FF7744")
        plt.bar(np.arange(8), values[:, 1], width=0.1, color="#5555FF")  #
        plt.scatter(5, values[5, 1], color='#D9006C', s=50, marker='*')
        plt.ylim((11, 12.2))
        plt.title("Results comparison about GD")

    plt.grid(axis='y')
    plt.legend()
    plt.xticks(np.arange(8), x_ticks, rotation=30)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel("Different Method")
    plt.ylabel("Values")
    plt.show()


def boxPlot(data, title):
    labels = ["Moving", "Elastix", "Sift", "SIRS-Net+", "FCC-Net"]

    colors = ['#CCCCFF', '#FFFFCC', '#CCFFFF', '#FFCCCC',
              '#FF9966']  # , "#CCCCFF"
    fig, axes = plt.subplots(nrows=1, ncols=1)
    p = axes.boxplot(x=data,
                     labels=labels,
                     notch=True,
                     patch_artist=True,
                     flierprops={
                         "marker": "o",
                         "markerfacecolor": "#FA7F6F",
                         "markersize": 8
                     })
    for patch, color in zip(p["boxes"], colors):
        patch.set_facecolor(color)
    axes.set_title(title)
    plt.show()


def violinPlot(data, title):
    labels = ["Moving", "Elastix", "Sift", "SIRS-Net+", "FCC-Net"]
    # , "#CCCCFF"
    colors = ['#CCCCFF', '#FFFFCC', '#CCFFFF', '#FFCCCC', '#FF9966']
    fig, axes = plt.subplots(nrows=1, ncols=1)
    p = axes.violinplot(data,
                        positions=range(0, 5, 1),
                        showmeans=True,
                        showmedians=False,
                        widths=0.8)

    # for patch, color in zip(p["bodies"], colors):
    #     patch.set_facecolor(color)
    #     patch.set_alpha(0.9)

    axes.set_xticks(range(0, 5, 1))
    axes.set_xticklabels(labels)
    axes.set_title(f"The results of {title}")

    # plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()


def densityPlot(data, title):
    plt.figure(figsize=(16, 10), dpi=80)
    labels = ["Moving", "Elastix", "Sift", "SIRS-Net+", "FCC-Net"]
    colors = ['#CCCCFF', '#FFFFCC', '#CCFFFF', '#FFCCCC', '#FF9966']
    sns.kdeplot(data[0], fill=True, color=colors[0], label=labels[0], alpha=.8)
    sns.kdeplot(data[1], fill=True, color=colors[1], label=labels[1], alpha=.8)
    sns.kdeplot(data[2], fill=True, color=colors[2], label=labels[2], alpha=.8)
    sns.kdeplot(data[3], fill=True, color=colors[3], label=labels[3], alpha=.8)
    sns.kdeplot(data[4], fill=True, color=colors[4], label=labels[4], alpha=.8)
    plt.title(f'Density Plot of {title}', fontsize=22)
    plt.legend()
    plt.show()


def getData(name='CC'):
    assert name in ["CC", "GC", "GD", "Dice"], "Not impletement"
    func = getattr(run, f"extract{name}")
    moving = func(loadlog("files/init.log"), 'm')
    elastix = func(loadlog("exp2/log/exp2_.log"), '0')
    sift = func(loadlog("exp2/log/exp2_.log"), '1')
    exp2 = func(loadlog("exp2/log/exp2_la2.5.log"), '2')
    exp3 = func(loadlog("exp3/log/cas3_corr.log"), '2')
    data = np.asarray([moving, elastix, sift, exp2, exp3])
    return data


def changeToDataFrame(name='CC'):
    data = getData(name)
    df = pd.DataFrame(data.T)
    df.columns = ["Moving", "Elastix", "Sift", "SIRS-Net+", "FCC-Net"]
    return df


def getExcel():
    dfs = []
    for i in ["CC", "GD", "Dice"]:
        dfs.append(changeToDataFrame(i))
    with pd.ExcelWriter("files/index.xlsx") as writer:
        for i, j in zip(["CC", "GD", "Dice"], dfs):
            j.to_excel(writer, sheet_name=i)


def plotRun(name='CC'):
    data = getData(name)
    # 箱线图
    # boxPlot(data.T, name)
    violinPlot(data.T, name)
    # densityPlot(data, name)


if __name__ == "__main__":

    # _5_3_diff(0)
    plotRun('Dice')
