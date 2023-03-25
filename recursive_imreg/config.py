import os


class Config:

    save_name = "recurse"

    root_path = 'Y:'

    # Dataloader Params
    numwork = 10

    # Networks Params
    epochs = 320
    save_epoch = 150
    lr = 1e-4  # 学习率
    step_size = 10
    gamma = 0.96  # 学习率衰减因子

    n_cascades = 1
    batch_size = 1
    channel = 8
    channel2 = 32

    def getinfo():
        return {
            "channel": Config.channel,
            "n_cascades": Config.n_cascades,
            "save_name": Config.save_name,
            "lr": Config.lr,
            "step_size": Config.step_size,
            "gamma": Config.gamma,
        }


if __name__ == "__main__":
    print(Config.getinfo())