""" Train a super-resolution model. """

import torch as th
import torch.nn.functional as F
from tools import logger
from tools.image_datasets import load_data
from diffusionModel.resample import create_named_schedule_sampler
from tools.script_util import sr_create_model_and_diffusion, load_config
from tools.train_util import TrainLoop


def main():
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("./public/configs/super_res_train.yaml")
    args_t = config['training']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logger.configure()

    # ------------ 扩散模型、神经网络、重要性采样的初始化 ------------
    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(args_m, args_d)
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args_t["schedule_sampler"], diffusion)

    # ------------ 数据集图像的预处理 ------------
    logger.log("creating data loader...")
    data = load_superres_data(
        args_t["data_dir"],
        args_t["batch_size"],
        large_size=args_m["large_size"],
        small_size=args_m["small_size"],
        class_cond=args_m["class_cond"],
    )

    # ------------ 开始走训练流程 ------------
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args_t["batch_size"],
        microbatch=args_t["microbatch"],
        lr=args_t["lr"],
        ema_rate=args_t["ema_rate"],
        log_interval=args_t["log_interval"],
        save_interval=args_t["save_interval"],
        resume_checkpoint=args_t["resume_checkpoint"],
        use_fp16=args_t["use_fp16"],
        fp16_scale_growth=args_t["fp16_scale_growth"],
        schedule_sampler=schedule_sampler,
        weight_decay=args_t["weight_decay"],
        lr_anneal_steps=args_t["lr_anneal_steps"],
    ).run_loop()


# 将原始的高分辨率图像加载器，转化为 “高分辨 + 低分辨” 的训练数据，以用于超分任务
def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False):
    # 加载高分辨率图像，返回的是一个 yield 生成器对象，
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
    )
    # 该对象包含了多个 图像内容 和 可选条件信息；
    for large_batch, model_kwargs in data:
        # 使用 interpolate 函数将 large_batch 下采样为 small_size 低分辨率图像，
        # 并保存在 model_kwargs["low_res"] 中，供模型作为条件输入。
        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        # 更新生成器对象
        yield large_batch, model_kwargs


if __name__ == "__main__":
    main()
