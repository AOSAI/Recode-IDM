"""
从图像样本 (image_sample.py) 给定的常规模型中生成一大批样本, 
再从超级分辨率模型中生成一大批样本。
"""

import os
import blobfile as bf
import numpy as np
import torch as th
from PIL import Image
from tools import logger
from tools.script_util import sr_create_model_and_diffusion, load_config
from diffusionModel.diffusion import SamplerDDPM


def main(sample_type):
    # ------------ 参数字典、硬件设备、日志文件的初始化 ------------
    config = load_config("../public/configs/image_sample.yaml")
    args_s = config['sampling']
    args_m = config['model']
    args_d = config['diffusion']
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    logger.configure()

    # ------------ 扩散模型、神经网络、低分辨率图像的初始化 ------------
    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(args_m, args_d)
    model.load_state_dict(th.load(args_s.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.log("loading data...")
    data = load_data_for_worker(args_s.base_samples, args_s.batch_size, args_m.class_cond)

    # ------------ 循环采样 ------------
    logger.log("creating samples...")
    all_images = []
    while len(all_images) * args_s["batch_size"] < args_s['num_samples']:
        model_kwargs = next(data)
        model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}

        # 调用 DDPM 的采样方法，生成一批图像
        sample = SamplerDDPM(diffusion).sample(
            model, args_m['large_size'],
            batch_size = args_s['batch_size'], 
            clip_denoised = args_s['clip_denoised'],
            model_kwargs = model_kwargs,
        )

        # 值域还原至 [0, 255]、形状变换至 [N, H, W, C]、确保张量是连续的
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # 超分输入的是模糊图，不依赖类别标签，所以只处理图像本身
        all_images.append(sample.cpu().numpy())
        logger.log(f"created {len(all_images) * args_s.batch_size} samples")

        # ------------ 保存类型：图像、numpy数组 ------------
    if "image" in sample_type:
        save_image(args_s, all_images)
    if "nparray" in sample_type:
        save_nparray(args_s, all_images)


def save_image(args_s, all_images):
    save_dir = os.path.join(logger.get_dir(), "samples")
    os.makedirs(save_dir, exist_ok=True)

    for i, img_arr in enumerate(all_images[: args_s["num_samples"]]):
        img = Image.fromarray(img_arr)
        filename = f"super_{i:06d}.png"
        img.save(os.path.join(save_dir, filename))

    logger.log(f"saved {args_s['num_samples']} images to {save_dir}")
    

def save_nparray(args_s, all_images):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args_s["num_samples"]]
    
    # 假设有 1000 次采样，shape_str = “1000x64x64x3”
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_super_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    
    np.savez(out_path, arr)
    logger.log("sampling complete")


def load_data_for_worker(base_samples, batch_size, class_cond):
    # base_samples 大概是一个 .npz 文件，加载 图像数组 和 对应标签
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]  # 低清图，形状为 [N, H, W, C]
        if class_cond:
            label_arr = obj["arr_1"]

    buffer = []  # 图像缓存数组
    label_buffer = []  # 标签缓存数组
    while True:
        for i in range(len(image_arr)):
            # 将 图像 和 标签，存入缓存数组
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            
            # 当数量满足一个 batch
            if len(buffer) == batch_size:
                # 把图像转换为 tensor，归一化到 [-1, 1]，通道顺序变为 [N, C, H, W]
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                
                # 构造一个 res 字典，第一个 key 为 low_res，值为处理过的图像张量 batch
                res = dict(low_res=batch)
                # 如果使用了标签条件，添加一个新的键值对
                if class_cond:
                    res["y"] = th.from_numpy(np.stack(label_buffer))
                
                yield res  # 返回 res 字典，供上层代码使用；
                buffer, label_buffer = [], []  # 清空缓存数组


if __name__ == "__main__":
    main(["image", "nparray"])
