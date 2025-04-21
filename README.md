## 1. 代码复现-IDM

该代码参考了 [improved-diffusion](https://github.com/openai/improved-diffusion) ，以软件工程的角度重构了代码的文件结构，同时写了 markdown 版的小白指南。欢迎各位道友用于学习，改造，和实验。

关键词：可学习方差、余弦噪声表、重要性采样、DDIM 加速采样

## 2. 环境配置

- system: win11
- conda 虚拟环境: python=3.12.0
- pytorch 版本: 2.4.1
- 其它相关依赖：查看 requirements.txt, 或代码中查看导入包的缺失

## 3. 目录结构

```py
Recode-IDM/
├── diffusionModel/           # 扩散模型核心
│   ├── noise_schedule.py       # β 线性时间表 及 相关计算
│   ├── diffusion.py            # q_sample, p_sample 等核心函数
│   └── utils.py                # 辅助函数
├── networkModel/             # 网络模型架构
│   ├── unet1.py                # U-Net + ResBlock
│   ├── unet2.py                # U-Net
│   ├── unet3.py                # NaiveConvNet
│   └── utils.py                # 辅助函数
├── scripts/                  # 训练采样入口
│   └── ts_unet1.py             # ts 表示 training and sampling
│   └── ts_unet2.py
│   └── ts_unet3.py
│   └── train.py                # 训练模型
│   └── sample.py               # 采样生成
├── documents/                # 每个文件的讲解笔记《小白指南》
│   └── *.md
├── public/                   # 公共资源
│   └── dataImg/                # 自定义的训练图像集
│   └── docsImg/                # markdown所需图像
├── main.py                   # 运行入口

├── jupyter/                  # Jupyter Notebook
    └── ddpm-1.ipynb            # B站大佬的 DDPM 简单入门实验
```

## 参考文献

[1. LittleNyima 大佬的博客 - 扩散模型（三）](https://littlenyima.github.io/posts/15-improved-denoising-diffusion-probabilistic-models/index.html)
