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

QKV 自注意力是 transformer 中的核心模块，具体的原理我找好了两个帖子，觉得写的还蛮容易理解的：

[1. transformer 中的 QKV 注意力机制 - 详细解读](https://zhuanlan.zhihu.com/p/414084879)
[2. transformer 中的 QKV 注意力机制 - 通俗理解](https://blog.csdn.net/Weary_PJ/article/details/123531732)

我在这里只做代码流程的说明，网络架构和残差块很类似，一个组归一化、一个三倍通道的卷积（对通道数做线性变换）、QKVAttention 的对象调用、输出层 proj_out 同样加了一个清零模型参数的外壳。

### 4.1 forward 函数

代码中的注释已经简明扼要的将用途说明了，这里只是做一些补充，比如：

```py
x = x.reshape(b, c, -1)  # == x.reshape(b, c, hw)
```

自注意力的本质是对序列中的每一个元素进行交互建模，要把空间拉直成“序列”。也就是把图像的空间展成一维序列，方便送进注意力。

```py
self.qkv = conv_nd(1, channels, channels * 3, 1)

qkv = self.qkv(self.norm(x))
qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
```

第一句话可真的是老朋友了，先对通道做归一化，一般用 LayerNorm 或者 GroupNorm，然后调用 self.qkv。这个卷积很有意思，它的卷积核为 1，等效于对 C 通道做线性变换，把 [b, c, hw] 映射为 [b, 3c, hw]，即三个分支 Q、K、V 的拼接。

多注意力头本质上相当于多加了几个分组的 batch size，会 reshape 成：[b * num_heads, head_dim * 3, hw]。这个 -1 我反正一开始是懵了，所以说明一下：

1. qkv.shape[2] 很明显就是 hw，假设它为 32+32=64.
2. b \* self.num_heads 假设 batch size 为 2，num_heads 为 4，这里就是 8.
3. 假设原始的通道数为 128，3 倍的 128 就是 384.

原始的总元素数量为：2 \* 384 \* 64 = 49152

新的 batch size 为 8，空间维度 hw 没有变，还是 64，那么通道数就变成了 49152 / （8\*64）= 96。

这里的 head_dim \* 3 只是一个称呼，和 QKVAttention 返回的 head_dim 是一个道理，就是说在有多个注意力头的情况下，单个的 QKV 分别占有多少通道。

之后的正儿八经的 qkv 的拆分和计算都是在 QKVAttention 中完成的，返回的 shape 应该是 [b * num_heads, head_dim, hw]，所以后面要接一个 reshape，把多个 head 的输出拼接回原通道维度（num_heads \* head_dim = c）。

最后的 proj_out 是另一个 1x1 卷积（或线性层），将每个位置的特征再调整一下。然后加上原始输入 x（残差连接），reshape 回 [b, c, h, w]，恢复图像结构。

### 4.2 QKVAttention 的 forward

这是标准的 QKV 自注意力机制的数学公式，看完应该比较容易理解 forward 中的代码：

<img src="../public/docsImg/qkv_attention.jpeg" width="360">

需要讲的重点有两个，一个是双开方，一个是 th.einsum 爱因斯坦求和约定。先对 einsum 的字母含义做一个说明：

| 字母 | 含义                                             |
| ---- | ------------------------------------------------ |
| b    | batch (其实这里是 batch 乘 num_heads)            |
| c    | 通道维度/每个 head 的维度                        |
| t    | query 的 token 序列位置，通常是空间位置，等于 hw |
| s    | key 的 token 序列位置，也是空间位置，等于 hw     |

==t 和 s 都是 token 的位置，只不过 t 是 query 的 index，s 是 key 的 index！==

第一个爱因斯坦求和约定计算："bct,bcs->bts"，可以这么理解：

```py
Q: [B, C, T]
K: [B, C, S] ← 这里 T = S = hw，但逻辑上一个是 query，一个是 key

# 注意力分数计算
weight[b, t, s] = sum over c (Q[b, c, t] * K[b, c, s])
```

👉 最终得到 [B, T, S]，即 每个 query token 对所有 key 的相关性矩阵。

第二个爱因斯坦求和约定计算："bts,bcs->bct", weight, v：

```py
# bts 是注意力权重 [B, T, S]
# v 是 value [B, C, S]
# 输出的结果为：
out[b, c, t] = sum over s (weight[b, t, s] * v[b, c, s])
```

也就是每个 query token 根据注意力权重对所有 value 进行加权求和，得到输出向量。

双开方其实是一个计算的小巧思，我们要计算 Q\*K / sqrt(ch)，而在 weight 的第一次计算中，使用的是 q _ scale 和 k _ scale 两个参数，这两个相乘之后，scale 相当于平方了一次，等于把双开方的一次开方抵消掉了。

巧妙的把除以 sqrt(ch) 换成了双开方的计算，在 q 和 k 的乘法计算前，提前分配给了 q 和 k，防止了内积变大造成 softmax 梯度消失的问题，提高了数值的稳定性。

### 4.3 QKVAttention 的 count_flops

这一段代码是给像 thop 这样的 FLOPs 计算工具用的。FLOPs 代表模型中需要做多少“乘加运算”，通常用来衡量模型的计算复杂度。原本的代码注释中也写了该怎么用（因为我的解释挪到这里了，所以就删除了）：

```py
macs, params = thop.profile(
    model,
    inputs=(inputs, timestamps),
    custom_ops={QKVAttention: QKVAttention.count_flops},
)
```

意思是：用 thop 来分析整个模型时，遇到 QKVAttention 这个模块，就调用它自定义的 count_flops() 方法来算。

## 5. U 型网络 UNetModel

### 5.1 标签条件扩散

首先第一个出现的新东西是，带分类的时间步嵌入：

```py
# init 构造函数中
if self.num_classes is not None:
    self.label_emb = nn.Embedding(num_classes, time_embed_dim)

# forward 实际执行函数中
if self.num_classes is not None:
    assert y.shape == (x.shape[0],)
    # 标签类别 + 时间步信息，形成条件向量
    emb = emb + self.label_emb(y)
```

nn.Embedding 方法的作用是，把类别标签（整数）映射成一个向量，向量的维度是 time_embed_dim，然后这个向量会被加到时间步嵌入上，一起送进模型，让 U-Net 学会“我现在要生成第 X 类的图片”。

🎯 为什么加到时间嵌入而不是图像上？因为 U-Net 的输入通常是纯噪声，还不知道是什么图像。而时间步嵌入是整个 U-Net 的调控器，把类别信息也加在它上面，效果稳定且容易实现。

### 5.2 识别时间步嵌入的容器

就是 unet 中第二个类，原本叫 TimestepEmbedSequential，这个名字真的是一股 java 味儿，不算差，但是我觉得太长了，我换成 TimeEmbedSeq 了。

它是干嘛用的呢？我们在 Resblock 残差块类中，用过 nn.Sequential 这个方法，就是在其中顺序添加各种模块，方便执行。而 TimeEmbedSeq 就是这个方法的加强版，它在顺序执行子模块时，会“识别”哪个子模块支持接收 timestep embedding（时间步嵌入 emb），然后：

1. 如果子模块是 TimestepBlock：layer(x, emb)，把时间步也传进去
2. 如果子模块不是 TimestepBlock：就普通前向 layer(x)

举个例子，input_blocks 中的第一个卷积层，它是纯粹的卷积模块，所以直接走 2；之后传入的模块，还有 ResBlock，以及 DownSample，前者继承的是 TimestepBlock 会走 1，后者没有时间步，会走 2.

🧠 为啥这么做而不是直接在 forward 里传不同参数？因为在训练中，我们会频繁堆叠很多 Block（有的要 emb，有的不要），写 if else 会很乱。所以作者用继承 + isinstance 判断的方式来自动分发，非常 Pythonic。

### 5.3 时间块的使用

有关残差网络和 U-Net 的结构，之前在 DDPM 中讲过，这里就不写了。我们可以看到，AttentionBlock 的使用是靠 attention_resolutions 参数和 ds 两个参数控制的。

这里的 ds 很有意思，在控制下采样的判断条件里，每一次下采样 ds 都会乘 2，我们知道 idm 设置的下采样通道倍率为 (1, 2, 4, 8)，而最后一次 8 它就不再下采样了，所以 ds 也是走 3 次，从 1 到 2、4、8。

也就是说，在这个条件判断语句中，它默认给的 attention_resolutions="16,8" 这个参数，只有 8 会生效：

```py
if ds in attention_resolutions:
    layers.append(
        AttentionBlock(
            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
        )
    )
```

### 5.4 forward 手动控制精度

```py
h = x.type(self.inner_dtype)
...
h = h.type(x.dtype)
```

默认情况下 pytorch 都是 float32 训练的，但是如果使用了混合精度训练（fp16， bf16）去优化部署，就得保证某些 数值不稳定的操作（比如 attention 权重的 softmax）始终用 float32 来算，不被自动降成 float16。

避免在 mixed-precision 训练下出现 梯度爆炸 / 收敛失败。比如，th.softmax(weight.float()) 中的计算就是先转换成 float32，再 type(weight.dtype) 转回去：

```py
weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
```

### 5.5 get_feature_vectors

它和 forward 函数长得很像，但它返回的是 result，是从 U-Net 中提取中间层的“特征向量”，而不是像 forward() 一样输出最终的噪声预测图（output image）。这么做通常是为了：

1. 作为一种 encoder，用于对输入图像做“语义嵌入”
2. 训练特征对齐、对比损失（contrastive loss）
3. 为 latent 模型做编码条件（condition）

它是设计者给研究人员留下的 hook 接口，目前在主流程中没有被用到，但它可以在你需要的时候调用，特别是在训练其他辅助任务或上层模块的时候。

## 6. 超分 SuperResModel

与 U-Net 网络一样，get_feature_vectors 只是备用接口，实现超分的还是 forward 函数。我们看看这三行共通的代码有什么用：

```py
# x 是高分辨率目标图像（可能是加噪后的），shape 还是 [B, C, H, W]
# 这行只是取出目标图像的尺寸，用于后面的对齐操作。
_, _, new_height, new_width = x.shape

# low_res 是对应的低分辨率图像（作为条件输入），比如下采样过的图。
# 这行代码将低分辨率图上采样回高分辨率，使其尺寸与 x 对齐。只是插值放大，并没有加入复杂的推理。
upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")

# 将目标图（x）和插值放大的低分辨率图在通道维度拼接。相当于：
# 我把图像本身和它的模糊版本一起输入 U-Net，让网络学会利用这个模糊图来更好地还原清晰图
x = th.cat([x, upsampled], dim=1)
```

最后再调用父类的 forward 函数进行训练。本质上它是一个映射模型：[x (有噪声的高分图) , 插值放大的 low_res 条件图] ——> 清晰的高分图像。

它不是靠这三行“实现”超分，而是靠整个 UNet 学习如何在有 low_res 条件输入的前提下生成更好的结果；这三行只是负责构造输入结构，告诉 UNet：“嘿，我现在想让你参考这个低分图，它是你可以依赖的上下文信息。”
