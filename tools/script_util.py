import yaml
from diffusionModel.respace import SpacedDiffusion, space_timesteps
from networkModel.unet import SuperResModel, UNetModel
from diffusionModel.noise_schedule import get_noise_schedule
from diffusionModel.utils import ModelMeanType, ModelVarType, LossType

NUM_CLASSES = 1000

def create_model_and_diffusion(model_kwargs, diffusion_kwargs):
    model = create_model(**model_kwargs)
    diffusion = create_gaussian_diffusion(**diffusion_kwargs)
    return model, diffusion

def sr_create_model_and_diffusion(model_kwargs, diffusion_kwargs):
    model = sr_create_model(**model_kwargs)
    diffusion = create_gaussian_diffusion(**diffusion_kwargs)
    return model, diffusion

# 通过图像尺寸得到 channel_mult 的最优解（base_channel为128）
def get_channel_mult(image_size: int):
    presets = {
        512: (0.5, 1, 1, 2, 2, 4, 4),
        256: (1, 1, 2, 2, 4, 4),
        128: (1, 1, 2, 3, 4),
        64: (1, 2, 3, 4),
        32: (1, 2, 2),
    }
    if image_size not in presets:
        raise ValueError(f"Unsupported image size: {image_size}")
    return presets[image_size]


def create_model(
    image_size, num_channels, num_res_blocks,
    learn_sigma, class_cond, use_checkpoint,
    attention_resolutions, num_heads, num_heads_upsample,
    use_scale_shift_norm, dropout,
):  
    # channel_mult 元组的（长度 - 1），决定下采样的次数；
    # 元组中的数值，决定通道数 base_channel 的变化情况
    channel_mult = get_channel_mult(image_size)

    # attention_resolutions 表示，你想在哪个图像尺寸中加入注意力机制，
    # 根据 channel_mult 的设定，每一种图像的尺寸最后都能缩放到 8 像素；
    # image_size // int(res) 表示尺寸的缩放倍数，和 unet 中的 ds 做对应。
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def sr_create_model(
    large_size, small_size,
    num_channels, num_res_blocks, learn_sigma,
    class_cond, use_checkpoint, attention_resolutions,
    num_heads, num_heads_upsample,
    use_scale_shift_norm, dropout,
):
    _ = small_size  # 假装使用了该变量，防止报错或滥用

    channel_mult = get_channel_mult(large_size)

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *, steps=1000, learn_sigma=False, sigma_small=False,
    noise_schedule="linear", use_kl=False,
    predict_xstart=False, rescale_timesteps=False,
    rescale_learned_sigmas=False, timestep_respacing="",
):
    
    betas = get_noise_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
        
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE

    if not predict_xstart:
        mean_type = ModelMeanType.EPSILON
    else:
        mean_type = ModelMeanType.START_X

    if not learn_sigma:
        if not sigma_small:
            var_type = ModelVarType.FIXED_LARGE
        else:
            var_type = ModelVarType.FIXED_SMALL
    else:
        var_type = ModelVarType.LEARNED_RANGE

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=mean_type,
        model_var_type=var_type,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

# 从 ymal 文件中获取参数字典
def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
