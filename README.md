## 1. 代码复现-IDM

该代码参考了 [improved-diffusion](https://github.com/openai/improved-diffusion) ，以软件工程的角度重构了代码的文件结构，有些地方有一点点的小修改，同时写了 markdown 版的小白指南。欢迎各位道友用于学习，改造，和实验。

Diffusion 关键词：可学习方差、余弦噪声表、重要性采样、DDIM 加速采样
U-Net 关键词：

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
│   ├── losses.py               # 损失函数的相关计算
│   └── utils.py                # 辅助函数
├── networkModel/             # 网络模型架构(神经网络)
│   ├── unet.py                 # U-Net + Resblock + Attention
│   ├── utils.py                # U-Net 相关的函数调用
│   ├── fp16_util.py            # 暂未优化
├── common/                   # 通用模块
│   ├── fp16_util.py            # 训练日志 setup
│   └── script_util.py          # ...
├── tools/                    # 项目级工具
│   ├── logger.py               # 训练日志 setup
│   └── script_util.py          # ...
├── scripts/                  # 训练采样脚本入口
│   ├── image_train.py          # 训练模型
│   └── image_sample.py         # 采样生成
├── documents/                # 每个文件的讲解笔记《小白指南》
│   └── *.md
├── public/                   # 公共资源
│   ├── configs/                # 参数配置文件
│   └── docsImg/                # markdown所需图像
├── main.py                   # 运行入口
├── jupyter/                  # Jupyter Notebook
```

## 参考文献

[1. LittleNyima 大佬的博客 - 扩散模型（三）](https://littlenyima.github.io/posts/15-improved-denoising-diffusion-probabilistic-models/index.html)

[2. transformer 中的 QKV 注意力机制 - 详细解读](https://zhuanlan.zhihu.com/p/414084879)

[3. transformer 中的 QKV 注意力机制 - 通俗理解](https://blog.csdn.net/Weary_PJ/article/details/123531732)
