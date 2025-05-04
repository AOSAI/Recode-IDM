不管是训练，还是采样，对于 神经网络模型、扩散模型 这两部分的参数基本都是一致的，变动的地方很少。只有采样/训练相关的参数变动较大。

## 1. 普通的训练和采样

### 1.1 神经网络 model

- image_size: 64
- num_channels: 128
- num_res_blocks: 2

这三个参数很好理解：图像的尺寸；神经网络的基础通道数；残差块的数量。

- num_heads: 4
- num_heads_upsample: -1
- attention_resolutions: "16,8"

前两个参数最开始挺误导人的，实际上是 QKV Attention 的时候才用到，分为几个头去实现注意力机制。而 attention_resolutions 同样是说，当下采样到什么地步开始嵌入 attention。

它在 unet 模型中，使用了一个 ds 参数来计数，但是因为采样缩放只有 (1, 2, 4, 8)，并且最后一次不再进行采样，所以 ds 最大只能到 8，默认的 16 属于无效参数。如果看不懂，请移步对应的文档和代码。

- learn_sigma: False
- class_cond: False
- use_checkpoint: False

learn_sigma 表示是否使用可学习方差，没有在 unet 网络中使用，而是在 script_util 中做了一个条件判断，如果为真使得 unet 的输出通道从 3 变成 6。

class_cond 表示是否进行分类的，扩散学习和采样。也是在 script_util 中使用，如果为真，便使用 NUM_CLASSES，表示区分多少个类别。

use_checkpoint 是梯度检查点机制，用于减少内存消耗。

- use_scale_shift_norm: True
- dropout: 0.1

use_scale_shift_norm 表示是否使用 FiLM 的方式进行时间步嵌入。原本是纯加法（平移），FILM 是缩放（乘法）+ 平移。

dropout 表示训练时，是否损失一部分比例的神经元，让模型拥有更好的泛化能力。

### 1.2 扩散模型 diffusion

- steps: 1000
- learn_sigma: true
- sigma_small: False

steps 表示扩散步数，learn_sigma 和上一部分一样，扩散模型和神经网络模型都需要这个参数，sigma_small 表示是否使用固定方差的 FIXED_SMALL。

- noise_schedule: "linear"
- use_kl: False
- predict_xstart: False

noise_schedule 加噪方式，分为 linear 和 cosine，use_kl 表示损失计算使用 RESCALED_KL，predict_xstart 表示预测的均值是否为 START_X，否的话均值预测的就是正常的噪声分布 EPSILON。

- rescale_timesteps: True
- rescale_learned_sigmas: True
- timestep_respacing: ""

rescale_timesteps

### 1.3 训练 training

### 1.4 采样 sampling

## 2. 超分的训练和采样
