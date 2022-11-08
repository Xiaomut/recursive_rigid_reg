import os


class Config:

    save_name = "recurse"

    root_path = '/home/wangs'

    # Dataloader Params
    norm = False  # 不能归一化
    numwork = 10

    # Networks Params
    epochs = 200
    save_epoch = 100
    lr = 1e-3  # 学习率
    step_size = 20
    gamma = 0.96  # 学习率衰减因子

    # Datasets Params
    testsize = 0.1  # 测试集比例

    n_cascades = 1
    batch_size = 2
    channel = 8

    def getinfo():
        return {
            "channel": Config.channel,
            "n_cascades": Config.n_cascades,
            "save_name": Config.save_name,
            "norm": Config.norm,
            "lr": Config.lr,
            "step_size": Config.step_size,
            "gamma": Config.gamma,
        }


if __name__ == "__main__":
    print(Config.getinfo())