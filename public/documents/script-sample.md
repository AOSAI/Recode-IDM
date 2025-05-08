## 1. image_sample

这一部分已经被我大刀阔斧的改版了。如果想要对比原版的 IDM，请自行下载，这里就不贴代码了。

首先是原本使用的 argparse 参数构造器，配合 script_util 文件中的默认参数定义列表，统一合成完整的参数字典。但是我觉得原本的方式，来来回回各种传参，既不方便查看参数，代码又显得很臃肿，所以我就使用了 yaml 的形式，直接定义成了字典。

其次，这个采样的原本用途是：从模型中生成一大批图像样本，并将其保存为一个大型 numpy 数组。这可用于生成用于 FID 评估的样本。但是很多时候，FID 这种数学对比它很抽象，我反正还是喜欢看图像的效果。

不过调用 diffusion 模块中的采样方法等步骤都是兼容的，所以我将保存方法分了两类，一种是保存成图像，一种是保存成 numpy 数组。当然了，两者可以同时选，方便建立实际效果与评估结果之间的联系。

### 1.1 主函数 main

循环采样简单说一下，all_images 用于保存所有采样出来的图像；all_labels 用于保存所有采样图像的类别，如果是类别条件生成的话。

调整数据格式部分，扩散模型输出的张量通常是 (B, C, H, W)，通道优先，值域在 [-1, 1]，而图像保存（PIL 或 numpy）通常使用的是 (H, W, C)，通道在最后，uint8，值域在 [0, 255]。

```py
sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
sample = sample.permute(0, 2, 3, 1)
sample = sample.contiguous()
all_images.extend(sample.cpu().numpy())
```

所以第一步，将模型输出从 [-1, 1] 映射到 [0, 255]。sample + 1 表示阈值范围从 [-1, 1] 变成 [0, 2]，然后 \* 127.5 就变成了 [0, 255], clamp(0, 255) 是为了防止越界的预防措施，最后转成 uint8，便于 Image.fromarray() 和 np.save。

第二步，从 (B, C, H, W) → (B, H, W, C)。

第三步，确保张量在内存中是连续的，避免 .numpy() 时报错（某些 permute 后的张量是 non-contiguous），这一步对语义没影响，但对内存布局很关键。

最后一步，相当于解构了一个维度的数组，将 (B, H, W, C) 变成了 (H, W, C) 的形式。

### 1.2 保存图像 save_image

这部分感觉没啥说的。为啥循环条件是这样的 all_images[: args_s["num_samples"]]，是因为采样次数是由 num_samples 决定的，但是它一个批次可能采样多张图像，所有最后 all_images 中的图像数量可能会超过 num_samples，所以用了切片的方式，保证保存的数量和 num_samples 对等。

### 1.3 保存数组 save_nparray

np.concatenate 的作用是，将所有图像 (H, W, C) 组合成 (N, H, W, C) 的形式，它是存储图像集合最标准的格式。

假设我们采样 1000 张图像，每一张是 64x64x3，如果直接 np.save(all_images)，保存的是一个 Python 对象数组（dtype=object），结构不规则，加载后可能还要手动解析，非常麻烦。

np.concatenate 会把这些 独立的图像变成一个标准的 4D array。这个结构非常适合：保存为 .npz 或 .npy、后续批量加载、批量可视化、用 NumPy、TensorFlow、PyTorch 的 Dataset 自动加载。

下一步的切片操作和 save_image 中的一个道理。标签的处理也是一样。

dist.get_rank() == 0，表示多卡训练或采样中，只让主进程保存文件，避免冲突。当然了，单卡也不会出错。

dist.barrier()就有意思了，因为只让主进程保存文件，非 0 的 rank 可能就会抢跑到后面的操作去了，所以需要控制大家一起等，防止有人偷跑。单卡不会出错。

## 2. super_res_sample

和 image_sample 基本是一样的，就是这里只保存图像，不保存标签。另外，重构输入的数据集函数 load_data_for_worker 已经在注释中讲解了。
