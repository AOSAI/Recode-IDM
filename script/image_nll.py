"""
评估扩散模型的 NLL (negative log-likelihood) 性能指标,
以每个像素的 bits-per-dimension (bpd) 来衡量生成图像的质量与模型的拟合能力。
"""

import os
import numpy as np
import torch as th
from tools import logger
from tools.image_datasets import load_data
from tools.script_util import create_model_and_diffusion, load_config


def main():
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("./public/configs/image_nll.yaml")
    args_s = config['sampling']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logger.configure()

    # ------------ 扩散模型、神经网络的初始化 ------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_m, args_d)
    model.load_state_dict(
        th.load(args_s["model_path"], map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # ------------ 数据集图像的预处理，测试、不打乱数据顺序 ------------
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args_s["data_dir"],
        batch_size=args_s["batch_size"],
        image_size=args_m["image_size"],
        class_cond=args_m["class_cond"],
        deterministic=True,
    )

    # ------------ 开始评估模型 ------------
    logger.log("evaluating...")
    run_bpd_evaluation(
        model, diffusion, device, data, args_s["num_samples"], args_s["clip_denoised"]
    )

# 对给定数量 num_samples 的图像进行 bpd 评估，包括：
# 变分下界 vb，像素重建误差 mse，原始的和预测的 x_0 之间的误差，总体的 bpd
def run_bpd_evaluation(model, diffusion, device, data, num_samples, clip_denoised):
    all_bpd = []
    all_metrics = {"vb": [], "mse": [], "xstart_mse": []}
    num_complete = 0

    while num_complete < num_samples:
        # 每次从 data 获取一个 batch；移动到 GPU；获取条件输入
        batch, model_kwargs = next(data)
        batch = batch.to(device)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}

        # 计算当前 batch 的 bpd 相关指标，返回值为字典
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # 计算 vb、mse、xstart_mse 的均值，并写入 all_metrics
        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].mean(dim=0)
            term_list.append(terms.detach().cpu().numpy())
        # 计算总体 bpd 的均值，并写入 all_bpd
        total_bpd = minibatch_metrics["total_bpd"].mean()
        all_bpd.append(total_bpd.item())
        
        num_complete += batch.shape[0]  # 循环条件更新
        logger.log(f"done {num_complete} samples: bpd={np.mean(all_bpd)}")  # 打印日志

    # all_metrics 中的三项是 bpd 的组成部分，主要是用来 研究、调参、画图 的
    # total_bpd 是这三项的综合，可以直接从 log 日志中看结果，没有后期可视化的必要
    for name, terms in all_metrics.items():
        out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
        logger.log(f"saving {name} terms to {out_path}")
        np.savez(out_path, np.mean(np.stack(terms), axis=0))

    logger.log("evaluation complete")
 

if __name__ == "__main__":
    main()
