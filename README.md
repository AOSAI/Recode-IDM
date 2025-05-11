## 1. 代码复现-IDM

该代码参考了 [improved-diffusion](https://github.com/openai/improved-diffusion) ，以软件工程的角度重构了代码的文件结构，有些地方有一点点的小修改，同时写了 markdown 版的小白指南。欢迎各位道友用于学习，改造，和实验。

Diffusion 关键词：可学习方差、余弦噪声表、重要性采样、DDIM 加速采样
U-Net 关键词：QKV 注意力机制、时间步嵌入方式、FP32 和 FP16 混合精度、超分

需要注意的是，我的文档是在我整理研究代码的时候写的，测试的时候又更改了一些小的细节，有错误。如果文档中看到对不上的地方，请按代码为主。

## 2. 环境配置

- system: win11
- conda 虚拟环境: python=3.12.0
- pytorch 版本: 2.4.1
- 其它相关依赖：查看 requirements.txt, 或代码中查看导入包的缺失

## 3. 目录结构

```py
Recode-IDM/
├── diffusionModel/           # 扩散模型核心(算法逻辑)
│   ├── noise_schedule.py       # beta 噪声时间表、相关计算
│   ├── diffusion.py            # 训练、原始采样、DDIM采样
│   ├── resample.py             # 重要性采样
│   ├── respace.py              # DDIM 的核心操作
│   └── utils.py                # 辅助函数
├── networkModel/             # 网络模型架构(神经网络)
│   ├── unet.py                 # U-Net + Resblock + Attention
│   ├── utils.py                # U-Net 相关的函数调用
├── tools/                    # 项目级工具
│   ├── image_datasets.py       # 训练图像的预处理文件
│   ├── logger.py               # 自定义日志工具
│   └── script_util.py          # 各种模型初始化的文件
│   └── train_util.py           # 训练模型的主函数
├── scripts/                  # 训练采样脚本入口
│   ├── image_nll.py            # 对数似然估计的模型评估
│   └── image_sample.py         # 采样生成
│   └── image_train.py          # 模型训练
│   └── super_res_*.py          # 超分的训练和采样
├── public/                   # 公共资源
│   ├── configs/                # 参数配置文件
│   └── docsImg/                # markdown所需图像
│   └── documents/*.md          # 每个文件的讲解笔记《小白指南》
│   └── jupyter/                # jupyter notebook
├── main.py                   # 运行入口
```

## 4. 数据集测试

两个数据集的来源是 github 的 [StarGAN v2 - Official PyTorch Implementation](https://github.com/clovaai/stargan-v2?tab=readme-ov-file#download-datasets)项目，大佬请自行处理。

### 4.1 CelebA-HQ 1024

CelebA-HQ 是一个高质量的人脸图像数据集，由 30,000 张分辨率为 1024×1024 的高质量 jpg 图像组成。[小白下载链接：CelebA-HQ 1024](https://www.dropbox.com/s/f7pvjij2xlpff59/celeba_hq.zip?dl=0)

### 4.2 Animal Faces-HQ （AFHQ）

AFHQ 是一个动物面孔的数据集，包含 15000 张左右，分辨率为 512×512 的高质量图像。该数据集包括猫、狗和野生动物三个域，每个域提供约 5000 张图像。其中每个域 500 张，总共 1500 张图像为测试集，其余的都为训练集。[小白下载链接：afhq-v2-dataset](https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0)

## 5. 参考文献

[1. LittleNyima 大佬的博客 - 扩散模型（三）](https://littlenyima.github.io/posts/15-improved-denoising-diffusion-probabilistic-models/index.html)

[2. transformer 中的 QKV 注意力机制 - 详细解读](https://zhuanlan.zhihu.com/p/414084879)

[3. transformer 中的 QKV 注意力机制 - 通俗理解](https://blog.csdn.net/Weary_PJ/article/details/123531732)
