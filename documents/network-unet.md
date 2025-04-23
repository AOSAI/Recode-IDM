## 上采样 Upsample

原始的 IDM 使用的是 ==最临近插值 + 卷积== 这样的组合方式，它的优势在于计算开销小，实现简单。代码如下：

```py
class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
```

但是相比于 ConvTranspose2d 还是略逊一筹，理由有三：

**1. 更强的表达能力**：ConvTranspose2d 是“可学习”的上采样，它通过反卷积实现上采样，同时引入了权重学习，适合用于生成模型中。
**2. 避免信息损失**：相比最近邻或双线性插值（+卷积）两步走，ConvTranspose2d 能在一步操作中完成放大和特征变换，减少可能的信息损耗。
**3. 结构更清晰简洁**：替换掉 interpolate + Conv2d 组合，可以让网络结构更统一、更便于控制输出大小和通道数。

需要注意的是，IDM 为了适配更多的任务，将不同类型、不同维度的网络层封装到了 nn.py，也就是现在的 utils.py 文件中。

## 下采样 Downsample

卷积网络层在 DDPM 中已经讲过，为什么卷积核是 3 而不是 4。这里只是多加了一个更为简单，但快速的平均池化网络层。

至于 3d 卷积的 stride 参数选择为什么是 (1, 2, 2)。看一下 2d 卷积的形状：[N, C, H, W]，再来看一下 3d 卷积的形状：[N, C, D, H, W]。

- N 代表同一个批次有多少张图片，也就是 batch size
- C 表示通道数，原始 DDPM 是 [3]，因为只预测噪声的均值；IDM 引入了可学习方差，所以是[6]，3 个均值，3 个方差
- D 深度，只在 3d 才有，可能表示 时间步、体积切片、视频帧 等等信息
- H 图像高度
- W 图像宽度

(1, 2, 2)就表示，D 不变，只对 H 和 W 做操作。也就是说它不在深度维（D）上缩小，只在空间维（H, W）上下采样。

##
