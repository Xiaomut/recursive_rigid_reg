## 级联网络

输入: `参考图像，浮动图像，GT_A，GT_B`
loss求 `pearson(参考图像 * GT_A, 浮动图像 * GT_B)`


## result

- `fullhis` 全图像直方图均衡后再截取作为输入，后面的通道数为 mid channel 1，默认8，结果8比16好 `FeatureRegression(mid_ch=mid_ch2)`

```py
nn.Sequential(Linear(3 * 4 * 4 * mid_ch, 256), LeakyReLU(0.1), Dropout(0.3), Linear(256, 6)) # mid_ch = 32
```
- `ori_corr` 原图截取作为输入，未作处理，结果比做好

```py
nn.Sequential(Linear(3 * 4 * 4 * mid_ch, 256), LeakyReLU(0.1), Dropout(0.3), Linear(256, 6)) # mid_ch = 32
```

- `ori_corr_low`原图截取作为输入，未作处理，网络结构改变。去掉了一层特征提取层，训练貌似变难了，效果不好
```py
# mid_ch = 6
nn.Sequential(LeakyReLU(0.1), 
    nn.Conv3d(320, mid_ch, 1, 1, 0),
    nn.InstanceNorm3d(mid_ch),
    LeakyReLU(0.1))
nn.Sequential(Linear(5 * 8 * 8 * mid_ch, 32), LeakyReLU(0.1), Dropout(0.3), Linear(32, 6))
```

- `ori_corr_high`原图截取作为输入，未作处理，网络结构改变。去掉了一层特征提取层，训练貌似变难了，效果不好
```py
# mid_ch = 6
nn.Sequential(LeakyReLU(0.1), 
    nn.Conv3d(8, mid_ch, 1, 1, 0),
    nn.InstanceNorm3d(mid_ch),
    LeakyReLU(0.1))
nn.Sequential(Linear(8 * mid_ch, 32), LeakyReLU(0.1), Dropout(0.3), Linear(32, 6))
```




## files

- `copy_image.py` 将第二部分临床数据进行数据格式化处理
- `hist_image.py` 将数据进行直方图匹配，用于得到分割标签

## bad segmentation

54 B 分割有问题
64 B 分割有问题
83 A 灰度差异很大
113 A 分割不是很好
116 A, B 灰度差异很大

## crop

45 坐标点靠下与靠前，目前没有做改动
53 坐标点靠下与靠前，目前没有做改动
84 平移差距大，未做更改
102 平移差距大，未做更改

## crop hisequal

3 有部分黑边
12 有部分干扰边？不确定
13 区别很大，有点问题
25 有部分干扰边
28 有部分干扰边
