"""
Train a diffusion model on images.
"""
import torch as th
from tools import logger
from tools.image_datasets import load_data
from diffusionModel.resample import create_named_schedule_sampler
from tools.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    load_config,
)
from common.train_util import TrainLoop


def main():
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("../public/configs/image_sample.yaml")
    args_t = config['training']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logger.configure()
        
    # ------------ 扩散模型、神经网络、重要性采样的初始化 ------------
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args_m, args_d)
    model.to(device)
    schedule_sampler = create_named_schedule_sampler(args_t.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args_t.data_dir,
        batch_size=args_t.batch_size,
        image_size=args_m.image_size,
        class_cond=args_m.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
