import numpy as np
import matplotlib.pyplot as plt


def plotCorr(feature):
    b, c, d, h, w = feature.size()
    feature = feature.view(b, c, d * h * w).squeeze(0).detach().numpy()
    plt.imshow(feature, cmap="gray")
    plt.title("F_9", loc="center")
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    pass