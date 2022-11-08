## 级联网络

输入: `参考图像，浮动图像，GT_A，GT_B`
loss求 `pearson(参考图像 * GT_A, 浮动图像 * GT_B)`

## files

- `copy_image.py` 将第二部分临床数据进行数据格式化处理
- `hist_image.py` 将数据进行直方图匹配，用于得到分割标签