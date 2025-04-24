## 1. 上采样 Upsample

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

## 2. 下采样 Downsample

卷积网络层在 DDPM 中已经讲过，为什么卷积核是 3 而不是 4。这里只是多加了一个更为简单，但快速的平均池化网络层。

至于 3d 卷积的 stride 参数选择为什么是 (1, 2, 2)。看一下 2d 卷积的形状：[N, C, H, W]，再来看一下 3d 卷积的形状：[N, C, D, H, W]。

- N 代表同一个批次有多少张图片，也就是 batch size
- C 表示通道数，原始 DDPM 是 [3]，因为只预测噪声的均值；IDM 引入了可学习方差，所以是[6]，3 个均值，3 个方差
- D 深度，只在 3d 才有，可能表示 时间步、体积切片、视频帧 等等信息
- H 图像高度
- W 图像宽度

(1, 2, 2)就表示，D 不变，只对 H 和 W 做操作。也就是说它不在深度维度（D）上缩小，只在空间维度（H, W）上下采样。

## 3. 残差块 ResBlock

我的 DDPM 是从 IDM 中拆解出来的，所以残差块中网络层的构建和现在的基本是一致的：

- in_layers：组归一化、SiLU 激活函数、卷积
- emb_layers：时间步嵌入
- out_layers：组归一化、SiLU 激活函数、Dropout 训练时是否丢弃一点神经元、卷积

需要说明的只有 emb_layers 和 out_layers。

### 3.1 时间步嵌入 emb_layers

时间步嵌入的通道数，用到了一个叫做 use_scale_shift_norm 的变量，如果为真，通道数翻倍，如果为假，通道数保持不变：

```py
2 * self.out_channels if use_scale_shift_norm else self.out_channels
```

这么写的原因是借鉴了 ==特征线性调制（Feature-wise Linear Modulation，FiLM）== 的原理，让时间步嵌入不只作用为“加法偏置”，而是可以同时对特征进行“缩放 + 平移”。在 forward 函数中是这么写的：

```py
# 原始的时间步嵌入为 加法偏置
h = h + emb_out

# FiLM 的时间步嵌入为 缩放（乘法） + 平移（加法）
# (1 + scale) 是为了初始化时更平滑，不会把特征直接变没。
h = out_norm(h) * (1 + scale) + shift
```

FiLM 提供了比“纯加法”更细粒度的控制，尤其适合控制时间步 t 不同特征的“风格”。我们看看 forward 函数中完整的 FiLM 计算：

```py
if self.use_scale_shift_norm:
    out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
    # torch.chunk 是 pytorch 中数组拆分的一个函数
    # 表示将某个数组(emb_out)，在哪个维度中(dim=1)，切分成几个(2)
    scale, shift = th.chunk(emb_out, 2, dim=1)
    h = out_norm(h) * (1 + scale) + shift
    h = out_rest(h)
```

self.out_layers 这个顺序模块中一共包含了四个步骤：[normalization, SiLU, Dropout, conv_nd]，先对其分了个组，out_norm 为 normalization，out_rest 为之后的三个。

第二行代码，是对时间步进行拆分。第三行代码就是 FiLM 版 时间步嵌入的计算；最后才接了 激活、卷积 等运算，实际的生成输出图像。

为什么这么安排？因为 scale/shift 操作是 直接作用在 norm(h) 上，激活、卷积等后续操作不能影响到缩放精度处理。

### 3.2 输出层结构 out_layers

不知道有没有人和我刚开始有一样的疑惑，如果给卷积网络包裹一个参数清零的函数，每次都清零了那还学习什么特征？

```py
self.out_layers = nn.Sequential(
    ...
    zero_module(
        conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
    ),
)
```

要弄清楚这一点首先要分清两点：

第一点，这里是残差块 ResBlock 类的构造函数（init），而构造函数中只是包含了模型的架构，对其进行初始化。

第二点，nn.Sequential 是一个用来组合各种神经网络模块的方法，本质上里面保存的所有结构，都是模块（nn.GroupNorm、nn.SiLU、nn.Dropout、以及 nn.Conv2d）。而 zero_module 是一个普通函数，初始化时调用一次就结束了，之后在 forward 函数中真正的调用的，只有那些模块。

如果觉得解释有点抽象，似懂非懂，可以看看 jupyter 文件夹中的 network 文件。

### 3.3 forward 函数与梯度检查点

这里使用了 ==梯度检查点== 方法。原本的函数是这样的，被它封装了一下：

```py
return checkpoint(
    self._forward, (x, emb), self.parameters(), self.use_checkpoint
)
```

首先是一个并没有被显式定义的参数 self.parameters()，它是 PyTorch 所有 nn.Module 子类自带的方法，它返回的是当前模块（以及其子模块）中所有需要梯度的参数。

记得 zero_module 函数中写了这么一句代码（和自身里直接用 self.parameters() 是一个道理）：

```py
for p in module.parameters():
```

之所以要传递它，这是 PyTorch 1.8+ 的 torch.utils.checkpoint 自定义实现中一个 trick：

通过把 parameters 也传进去，能够保证这些参数在计算图中被追踪（即使在被 checkpoint 包裹的函数中）；否则，如果你只传了 x 和 emb，而这些参数在 self.\_forward 中是隐式用到的，PyTorch 有可能会丢失对这些权重梯度的追踪！

但是我觉得并没有简单的条件语句看着舒服，所以我将原本的写法改掉，把原本的 checkpoint 函数也删除掉了，直接调用 CheckpointFunction 函数。

需要说明的是，**调用 .apply() 方法时**，PyTorch 内部会自动调用 CheckpointFunction.forward()，等到主循环中计算反向传播 loss.backward() 的时候，会直接调用 CheckpointFunction.backward() 方法。不用我们手动的去选择 forward 和 backward。

### 3.4 继承 TimestepBlock

TimestepBlock 是继承了 nn.Module 的一个抽象基类，原本的 forward 方法只有一个 x 作为参数，但是抽象基类 TimestepBlock 将其转变为了 x 和 emb 两个参数，方便了 残差块 forward 函数的使用。

## 4. 注意力机制 AttentionBlock
