## 进行所有模型评估

1. 首先处理部分分割文件问题 `filepro`
    - 1. `move_elastix` 由 `MEL-Net` 分割完成，需要移动文件
    - 2. `sift_seg` 通过浮动图像以及sift得到的label进行变形
完成准备工作

2. 实验有三部分
