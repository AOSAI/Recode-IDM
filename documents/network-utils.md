## 网络层组件

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
