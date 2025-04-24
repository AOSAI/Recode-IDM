## 1. 网络层组件

- conv_transpose_nd：逆卷积模块，用于上采样
- conv_nd：卷积模块，用于下采样
- avg_pool_nd：平均池化模块，用于下采样
- linear：线性模块，用于对齐图像和时间步的通道数

```py
def conv_transpose_nd(dim, *args, **kwargs):
    return nn.ConvTranspose2d(*args, **kwargs)
```

几乎所有模块都用到了这个写法，他是 Python 中的参数打包机制：

1. \*args 是接收所有位置参数（比如 foo(1, 2, 3)）
2. \*\*kwargs 是接收所有关键字参数（比如 foo(x=1, y=2)）

拿上采样的写法举例：

```py
# *args 就相当于 channels, channels
# **kwargs 相当于 kernel_size=4, stride=2, padding=1
conv_transpose_nd(dims, channels, channels, kernel_size=4, stride=2, padding=1)
```

## 2. 组归一化

在之前我都是直接调用的 nn.GroupNorm(32, in_channels)，但是 IDM 它继承并重写了 nn.GroupNorm 中的 forward 方法，一共做了两件事：

```py
return super().forward(x.float()).type(x.dtype)
```

- 把 x 转成 .float()（也就是 float32）
- forward 之后再 .type_as(x) 转回来（比如原来是 float16 就转回 float16）

这么做主要是为了处理 自动混合精度训练（AMP）带来的坑。GroupNorm 在低精度（如 float16）下容易出现数值不准、不收敛、爆 NaN 等问题，特别是小 batch size 或少通道的时候。

至于这里的封装是为了预防，还是真的已经有了混合精度运算，先打个 tag。

## 3. 梯度检查点机制

IDM 所使用的 CheckpointFunction 函数，继承的是 th.autograd.Function，是一个更底层的实现，可以在反向传播中控制哪些操作不被保存，并通过自定义的方式手动计算梯度。这种方法提供了更多的灵活性，但也相对复杂。

更简单的方式是使用简化的高层 API：torch.utils.checkpoint()，一个小对比：

| 特性     | torch.utils.checkpoint       | th.autograd.Function                           |
| -------- | ---------------------------- | ---------------------------------------------- |
| 实现层级 | 高层 API（简化）             | 自定义的低层实现                               |
| 控制粒度 | 自动控制分段和存储           | 手动控制每一步的前向和反向                     |
| 用途     | 一般适用于节省内存、通用任务 | 适用于特殊的自定义需求，需要手动实现前向和反向 |

### 3.1 注解 @staticmethod

静态方法仅仅绑定类中的某一个方法或者属性，它们不需要创建类的实例，因此不依赖于对象的状态。静态方法通常用于实现一些与类相关但不需要访问类属性或实例属性的功能。

### 3.2 自定义 forward

| 步骤                  | 内容                             | 目的                                |
| --------------------- | -------------------------------- | ----------------------------------- |
| ctx.run_function      | 保存要执行的函数                 | 后面 backward 会再用到这个函数      |
| ctx.input_tensors     | 把前 length 个参数保存为输入数据 | 这些是需要参与反向传播的 input      |
| ctx.input_params      | 把后面的参数保存为模型参数       | 它们的 .grad 会被正确追踪           |
| with torch.no_grad(): | 关闭 autograd                    | ❗ 前向时不记录中间计算图，节省显存 |
| output_tensors        | 调用实际函数算结果               | 此时 output 没有梯度历史            |

这就是 gradient checkpointing 的核心原理：前向传播时不存中间激活，等到反向传播时才重新计算。所以这个 forward() 是 **假的**前向传播，为了节省内存：只把结果算出来用着，不保存中间梯度历史。

就好像考试时先把题目答案记下来，过程不写，等老师问你“怎么得出来的？”，你再临时补过程 😅。

### 3.3 自定义 backward

这一部分就是，老师真的问你，你是怎么得出这个结果的？学渣唯唯诺诺，而学霸已经开始计算了：「再来一次真前向 + 真反向」

✅ 第一步：准备重新计算的输入 ctx.input_tensors：

把保存下来的输入（ctx.input_tensors）重新 requires_grad=True，因为前向时用了 torch.no_grad()，要重新启用梯度追踪。

✅ 第二步：小技巧 - 复制一下 shallow_copies：

防止 run_function 里有 inplace 操作（修改了输入的存储），.view_as(x) 不会分配新内存，但可以规避这种操作对原始张量的影响，有些模型写得“野”，就容易碰上这个坑。

✅ 第三步：重新做一次前向传播！output_tensors：

因为第二步和第三步写明了，with torch.enable_grad()，表示已经开了梯度追踪了，所以这一次才是真·有梯度的 forward。

✅ 第四步：计算反向传播 input_grads：

现在拿到前向输出 output_tensors 和输出梯度 output_grads，求输入变量 input_tensors + 模型参数 input_params 的梯度。

allow_unused=True 表示，如果有些输入变量在计算图中没有参与到输出的计算中，那么就返回 None，不要报错。在 checkpoint 中，有时候因为重计算的路径不是完整的一遍，所以可能有中间变量没参与到最后输出，因此要容忍这种“暂时没用到”的情况。

✅ 第五步：释放显存 & 返回：

del 表示清理内存，而 return 的格式中，None, None 表示 run_function, length 没有梯度，input_grads 是返回给 x 和 params 的梯度。
