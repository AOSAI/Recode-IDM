这一部分就是==重要性采样==，是一个优化训练效率和梯度估计质量的关键设计。在训练 diffusion model 时，我们每次都要对一个随机的 t 来训练，但是有的时间步（比如 t 很大或很小）训练贡献小，有的时间步 loss 震荡大、梯度有用。所以我们可以用不同的采样策略对时间步 t 进行“非均匀采样”，来提升训练效果和效率。

![](../public/docsImg/importance_sampling-1.jpeg)

## 1. create_named_schedule_sampler

训练 py 文件调用重要性采样的入口，一共有两个选择。

## 2. ScheduleSampler

是所有重要性采样对象的父类。定义了一个权重函数 weights，所有子类必须实现，定义了一个 sample，这里面实现了 UniformSampler 的采样。

首先 w 就是 UniformSampler 传过来的 np.ones([diffusion.num_timesteps])，将所有的时间步的权重都定义为 1，也就是所有权重都一样。然后 p 是每个时间步的概率。

np.random.choice 一共有四个参数：

- len(p) 就是有多少个 t，P 的数量和 t 是一致的，而 t 又是连续的正整数，所以这就代表了 timesteps
- size=(batch_size,)，表示从 len(p) 中取 batch_size 数量个值
- replace=True，这里没写是因为默认为 true，表示是否可以取相同的元素
- p=p，这个表示用 p 的概率去获取 timesteps

下一步就是从 np 数组转换为 tensor 张量。

计算反权重，假设我们扩散过程有 1000 步，每一步的权重都为 1，那么每一步的概率也是相等的，都为 1/1000，所以采样出来的新的扩散时间步的概率 indices_np，仍旧是 1000 个 1/1000。

那么在计算 weights_np 的时候，len(p) \* p[indices_np] 等于恢复原本的权重，但是最后做了一个倒数，这才是新权重更新的重点，不过由于原来就是 1，倒数了还是 1，所以默认的均匀权重，是没有变化的。

### 2.1 UniformSampler

没有内容，虚无。

## 3. LossAwareSampler

```py
ScheduleSampler         # 抽象基类，定义接口：sample(), weights()
│
├── UniformSampler      # 实现了 sample() 和 weights(): 全是均匀的 1
│
└── LossAwareSampler    # 抽象类，只实现了 loss 更新机制，自己不定义 weights()
     │
     └── LossSecondMomentResampler  # 核心！实现了基于 loss 的权重策略（重点）
```

原本的结构是这样的，但是 LossAwareSampler 本质上只实现了一个 update_with_local_losses 方法，并且这个方法都是对多卡训练的处理，所以我直接删除了 LossAwareSampler，让 LossSecondMomentResampler 直接继承 ScheduleSampler。

## 4. LossSecondMomentResampler

参数 history_per_term 表示每个时间步，需要保留多少个历史值；参数 uniform_prob 表示一个平滑项，避免某些 timestep 权重为零（类似 label smoothing），在 weights 方差中被使用。

参数 \_loss_history 和 \_loss_counts 已经在注释里面解释过了。需要注意的是 \_loss_counts，这个形状虽然是 timesteps，但是表示的是 \_loss_history 中已经记录的历史值的索引，它仅仅只是一个计数器，从 0 到 10。

### 4.1 weights

它有两种形态，如果所有的时间步，还没有完全累计满十个历史值，那么就返回并使用平均权重。满足了的话，**重要性度量**部分：

```py
weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
```

✅ 第一步：计算 loss 的二阶矩（这就是“loss 的波动程度”或“幅度大小”，可以看成是该时间步 loss 的重要性度量。）

1. self.\_loss_history: 是一个形状为 [T, K] 的数组，表示 T 个 timestep，每个有 K 个历史 loss
2. self.\_loss_history \*\* 2: 对所有 loss 值求平方
3. np.mean(..., axis=-1): 对每一行（一个时间步的 K 个 loss）求平均平方
4. np.sqrt(...): 相当于求 均方根（RMS）

```py
weights /= np.sum(weights)
```

✅ 第二步：标准化为概率分布（加权采样的“权重分布”）

把所有 timestep 的权重加起来，每个除以总和 → 得到一个概率分布（总和为 1）

```py
weights *= 1 - self.uniform_prob
weights += self.uniform_prob / len(weights)
```

✅ 第三步：加入均匀采样的保底策略（不能把所有概率都集中在几个 loss 高的时间步上，否则训练不稳定。这段代码的目标是“加一点随机性”，防止采样分布太极端）

- uniform_prob: 是一个很小的值（比如 0.001）
- (1 - uniform_prob)：让 99.9% 的概率来自你的 loss 加权
- uniform_prob / len(weights)：剩下 0.1% 均匀分配给所有 timestep，确保每个 timestep 都有机会被采样

### 4.2 update_with_local_losses

这个函数原本是写在 LossAwareSampler 中的，实际上只在多卡训练时有用，而且是为了配合：dist.all_gather(...)，但是我只用单卡，所以没有一点暖用，直接删了。

### 4.3 update_with_all_losses

- ts: 是一个列表，比如 [15, 99, 200, 300]，表示这四个样本分别采样自哪几个 timestep。
- losses: 是每个样本对应的 loss 值，比如 [0.32, 0.17, 0.43, 0.29]

我们想要做的事情是，把这些 loss 分别更新到对应的 timestep 的历史 loss 列表中。所以就自然写成了：

```py
for t, loss in zip(ts, losses):
```
