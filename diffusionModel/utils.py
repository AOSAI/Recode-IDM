import torch as th
import numpy as np
import enum

class ModelMeanType(enum.Enum):
    """决定模型预测的内容是什么，用于训练"""
    PREVIOUS_X = enum.auto()    # 预测 x_{t-1}
    START_X = enum.auto()       # 预测 x_0
    EPSILON = enum.auto()       # 预测噪声 epsilon    

class ModelVarType(enum.Enum):
    """决定反向过程中协方差的计算方式，用于采样而非训练"""
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()   # 固定小方差，betas
    FIXED_LARGE = enum.auto()   # 固定大方差，posterior_variance
    LEARNED_RANGE = enum.auto() # 

class LossType(enum.Enum):
    """模型的损失计算方式"""
    MSE = enum.auto()       # 使用原始 MSE 损失（学习方差时使用 KL）
    RESCALED_MSE = enum.auto()  # 使用原始 MSE 损失（学习方差时使用 RESCALED_KL）
    KL = enum.auto()        # 使用变分下界 (variational lower-bound)
    RESCALED_KL = enum.auto()   # 与 KL 相似，但要重新缩放以估算整个 VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL

def extract(arr, timesteps, broadcast_shape):
    """
    从 1D numpy 数组中按时间步采样，并 broadcast 到目标形状。

    :param arr: 1D numpy 数组 (如 beta schedule)
    :param timesteps: shape 为 [batch_size] 的时间步张量。
    :param broadcast_shape: 用于广播的目标形状（如 [B, 1, 1, 1]）。
    :return: 已广播的张量, shape 为 broadcast_shape。
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)