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

- data_dir: ""
- schedule_sampler: "uniform"

这两个值在送入 TrainLoop 之前已经被调用，前者是**数据集预处理**，返回的是 data，后者是**重要性采样**，返回的是初始化好的对象。两个返回值被送入 TrainLoop 中。

- lr: 1e-4
- weight_decay: 0.0
- lr_anneal_steps: 0

lr 表示**学习率**，1e-4 是科学计数法，等于 0.0001（1 乘以 10 的负 4 次方）。weight_decay 表示**权重衰减**，防止模型过拟合，是参数构造器中的参数。lr_anneal_steps 代表**学习率衰减步数**，控制学习率从初始值逐步衰减到 0。

- batch_size: 32
- microbatch: -1
- ema_rate: "0.9999"

batch_size 就是同一批次训练图像的数量，根据我之前的经验，最佳范围应该是 16 到 64。microbatch 表示最小的 batch_size。ema_rate 表示**指数滑动平均率**，对模型权重进行平滑，用于生成更稳定、更泛化能力强的模型。

- log_interval: 10
- save_interval: 10000
- resume_checkpoint: ""

log_interval 表示训练多少轮，输出一次 logger（txt 文件以及控制台）。save_interval 表示训练多少轮保存一次模型的参数（pt 文件）。resume_checkpoint 表示是否有训练好的 pt 文件，有的话在此基础上进行训练。

- use_fp16: False
- fp16_scale_growth: 1e-3

use_fp16 代表是否使用 float16 这个精度。fp16_scale_growth 表示**半精度缩放因子增长**，用于动态调整 loss_scaler，防止使用 FP16 训练时出现数值下溢。

### 1.4 采样 sampling

- clip_denoised: True
- num_samples: 10000

num_samples 代表实际想要采样多少张图片；clip_denoised 它控制是否对模型预测的图像 x_start 进行 像素范围裁剪，通常是裁剪到 [-1, 1] 区间。

✅ 为什么要裁剪？

1. 防止像素爆炸：部分时间步（尤其是早期）模型预测不准，x_0 可能超出值域 [-1, 1]，导致后续计算时出现异常放大。
2. 提高生成图像质量：可避免强噪点或局部伪影。
3. 最后，如果训练中用过 clip_denoised=True，采样时也建议打开它。保持训练-测试一致性。

- batch_size: 16
- use_ddim: False
- model_path: ""

## 2. 超分的训练和采样

### 2.1 采样

model 部分的参数 image_size 变成了 large_size 和 small_size，前者为高分辨率图像尺寸，后者为低分辨率图像尺寸。

sample 部分的参数多了一个 base_samples 参数，是一个 .npz 文件 的路径。

### 2.2 训练

训练部分只换了一个 image_size，和采样一样，其他地方都没有变动。

## 3. 负对数似然估计 NLL

只有 sampling 这里换了几个参数，去掉了 use_ddim，新增了一个 data_dir，这个是图像数据集的路径。
