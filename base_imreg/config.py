import os

is_linux = os.path.exists('/home/wangs')


class Config:

    save_name = "result"
    image_mode = "B2A"

    root_path = '/home/wangs' if is_linux else 'Y:'
    result_path = f'{save_name}/result'

    # Dataloader Params
    norm = True
    change = True
    batch_size = 2
    numwork = 15 if is_linux else 0

    # Networks Params
    midch = 8  # mid channel
    growthrate = 4
    save_epoch = 100
    epochs = 200
    lr = 2e-4  # 学习率
    step_size = 30  # 学习率衰减步长
    gamma = 0.9  # 学习率衰减因子

    # Datasets Params
    testsize = 0.1  # 测试集比例

    # Adam or SGD Params
    b1 = 0.9
    b2 = 0.999
    mome = 0.9  # 动量参数
    decay = 1e-6  # L2衰减率
