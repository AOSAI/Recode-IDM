在 Python 中，类与类之间存在三种关系：依赖关系、组合关系、继承关系。IDM 使用了两种方式去做逆向生成的采样，一种是原始 DDPM，一种是加速采样的 DDIM。

我将两种不同的采样方法，定义在了新类 SamplerDDPM 和 SamplerDDIM 中，并用了组合的方式，将基类 GaussianDiffusion 封装到了两个采样对象的属性中。这样看起来结构清晰，并且未来拓展也变得更容易。

## 1. GaussianDiffusion

在这个基类对象中，我只保留了构造函数（beta 相关计算参数、传入的参数），和扩散、训练相关的函数。一些内容和 DDPM 一致，就不重复讲了，这里仅将不同的地方做一个说明。

### 1.1 q_mean_variance

这个函数是用来计算，在前向加噪过程中，真正的分布的计算。直觉上的理解：均值就是原始图像保留下的部分，方差就是逐步加入的高斯噪声。对比真实后验分布：

- q_mean_variance: 我如何丢失信息”，是可以直接计算出来的，不依赖模型。
- q_posterior: “我如何一步步找回信息”，依赖模型预测的噪声，以及从噪声推断出的 x_0。

### 1.2 p_mean_variance 方差

==首先说，FIXED_LARGE 和 FIXED_SMALL 这两个方差选择==。在我写的 DDPM 的构造中，FIXED_LARGE 代表的是真实后验方差，FIXED_SMALL 代表的是 betas 构建的方差。而 IDM 的定义反过来了，为什么？

从方差数组之间的间隔来说，betas 所代表的数组确实要比后验方差数组要密集，要小。但是 IDM 是从采样的多样性角度来分类的。在 DDPM 的采样步骤 p_sample 中，每一步我们都做类似这样的操作：

```py
# noise ~ N(0, I) 是随机采样的。
x_{t-1} = mean + sqrt(var) * noise
```

额外注入的噪声，是跟方差成正比的。所以本质上，多样性 == 采样噪声的影响力。

```py
# ➤ 使用 posterior_variance 时：
x_{t-1} = E[q(x_{t-1}|x_t, x_0)] + sqrt(posterior_var) * N(0, I)
```

这个 posterior_var 是严格根据 x_0 推导出来的真实分布。这时候采样基本“贴着真实轨迹走”。所以结果很稳定 → 生成图像几乎不变，多样性低。

```py
# ➤ 使用 beta_t（更小的 var）时：
x_{t-1} = E[...] + sqrt(beta_t) * N(0, I)
```

虽然 beta 比 posterior_var 小，但它跟采样噪声是独立组合的！它不是推导出来的真实后验，只是人为固定个数值，允许噪声多发挥一点作用。所以虽然 sqrt(beta) 更小，但整个过程和原始 q 不一致，会累积出不同的轨迹。

==其次就是 IDM 的一个重点，可学习方差 LEARNED 和 LEARNED_RANGE==。

### 1.3 p_mean_variance 均值
