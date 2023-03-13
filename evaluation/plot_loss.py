import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

plt.rc('font', family='Times New Roman')


def _5_3_diff():
    values = np.asarray([[0.8165, 12.1183, 0.8865], [0.845, 11.7894, 0.9052],
                         [0.8428, 11.591, 0.9025], [0.8565, 11.5008, 0.9085],
                         [0.8536, 11.4475, 0.9067], [0.861, 11.2416, 0.913],
                         [0.8306, 11.5496, 0.89], [0.8407, 11.6818, 0.8974]])

    x_ticks = [
        i.replace(' ', '') for i in [
            "n=1", "Corr, n=1", "n = 2", "Corr, n = 2", "n = 3", "Corr, n = 3",
            "n = 4", "Corr, n = 4"
        ]
    ]
    # plt.plot(np.arange(8), values[:, 0], label="NCC")

    # plt.scatter(5, values[5, 0], color='#D9006C', s=50, marker='o')
    # plt.plot(np.arange(8), values[:, 2], label="Dice")
    # plt.scatter(5, values[5, 2], color='#D9006C', s=50, marker='o')
    plt.plot(np.arange(8), values[:, 1], label="GD")
    plt.scatter(5, values[5, 1], color='#D9006C', s=50, marker='o')

    plt.legend()
    plt.grid()

    plt.xticks(np.arange(8), x_ticks, rotation=30)
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel("Different Method")
    plt.ylabel("Values")
    plt.title("Results comparison about GD")

    plt.show()


def boxPlot(data, title):
    labels = ["Moving", "Sift", "Elastix", "Ants", "SSC-Reg", "SML-Reg"]

    colors = ['#CCCCFF', '#FFFFCC', '#CCFFFF', '#FFCCCC', '#FF9966', "#CCCCFF"]
    fig, axes = plt.subplots(nrows=1, ncols=1)
    p = axes.boxplot(
        x=data,
        labels=labels,
        notch=True,
        patch_artist=True,
        medianprops={
            'color': '#8ECFC9',
            'linewidth': '1.5'
        },
        #  meanline=True,
        #  showmeans=True,
        meanprops={
            'color': '#FFBE7A',
            'ls': '--',
            'linewidth': '1.5'
        },
        flierprops={
            "marker": "o",
            "markerfacecolor": "#FA7F6F",
            "markersize": 10
        })
    for patch, color in zip(p["boxes"], colors):
        patch.set_facecolor(color)
    axes.set_title(title)
    plt.show()


def plotNCC():
    ev = Evaluation()
    ncc_mean, ncc_std = ev.processNcc()
    ncc = np.asarray([
        ev.moving_ncc, ev.sift_ncc, ev.elastix_ncc, ev.ants_ncc, ev.p3_ncc,
        ev.ours_ncc
    ],
                     dtype=np.float)
    boxPlot(ncc.T, title="NCC")


def plotGD():
    ev = Evaluation(cal_ncc=False, cal_ssim=False, cal_nmi=False)
    gd_mean, gd_std = ev.processGD()
    gd = np.asarray([
        ev.moving_gd, ev.sift_gd, ev.elastix_gd, ev.ants_gd, ev.p3_gd,
        ev.ours_gd
    ],
                    dtype=np.float32)
    boxPlot(gd.T, title="GD")


if __name__ == "__main__":

    _5_3_diff()