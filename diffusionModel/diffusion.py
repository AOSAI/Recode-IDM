import torch
import torch.nn as nn
import numpy as np
from .noise_schedule import get_noise_schedule
from .utils import ModelMeanType, ModelVarType, LossType, extract
from .losses import normal_kl, approx_standard_normal_cdf, discretized_gaussian_log_likelihood

# -------------- 基类 --------------
class GaussianDiffusion:
    def __init__(self, model, model_mean_type, model_var_type, loss_type, 
                 timesteps=1000, device="cuda"):
        # 将传入的参数注册为 self 全局变量
        self.model = model
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.timesteps = timesteps
        self.device = device

        # 获取所有噪声调度参数，并注册为 self.xxx
        noise_schedule = get_noise_schedule(timesteps)
        for k, v in noise_schedule.items():
            setattr(self, k, v)

    def q_sample(self, x_start, t, noise=None):
        """从原始图像添加噪声, 模拟前向扩散过程的采样"""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):
        """计算真实后验的均值和方差, 用于采样下一个时间步骤 x_(t-1)"""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def q_mean_variance(self, x_start, t):
        """向前加噪分布的直接表达式, 辅助函数, 在VLB计算中使用"""
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_x0_from_eps(self, x_t, t, noise):
        """根据模型预测的噪声 ε，反推出原始图像 x_0 的估计值"""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def predict_x0_from_xprev(self, x_t, t, xprev):
        """根据 x_t 和 x_(t-1)，反推 x_0 的估计值"""
        return (  # (xprev - coef2 * x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev - 
            extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape)
            * x_t
        )
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """返回模型预测的噪声均值、方差（固定、可学习）"""
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2] 
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # --------------- 可学习方差，通道数加倍 ---------------
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
                max_log = extract(torch.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
            # --------------- 固定方差 ---------------
            if self.model_var_type == ModelVarType.FIXED_LARGE:
                model_variance = np.append(self.posterior_variance[1], self.betas[1:])
                model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
            else:
                model_variance = self.posterior_variance
                model_log_variance = self.posterior_log_variance_clipped

            model_variance = extract(model_variance, t, x.shape)
            model_log_variance = extract(model_log_variance, t, x.shape)
        else:
            raise NotImplementedError(f"Unknown model_var_type: {self.model_var_type}")

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self.predict_x0_from_xprev(x, t, model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self.predict_x0_from_eps(x, t, model_output)
                )
            model_mean, _, _ = self.q_posterior(pred_xstart, x, t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }
    
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        估算变分下界 (Variational Lower Bound, VLB) 的每一步的贡献,
        非训练用损失, 而是用来衡量 模型生成效果是否符合理论 的一个指标,
        通常在 validation 或 debug 阶段用来算 Bits-Per-Dimension (bpd)。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start, x_t, t)
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """损失函数调用入口, 用于计算单个时间步下的损失值"""
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        term = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            term["loss"] = self.vb
    
    def p_losses(self, model, x_start, t, noise=None):
        """损失函数调用入口, 用于计算单个时间步下的损失值"""
        if self.loss_type == LossType.MSE:
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError("Only MSE loss is supported for now.")
        
        t = t.to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)  # 加噪音得到 x_t        
        predicted_noise = model(x_t, t)  # 模型预测噪音

        return self.loss_fn(predicted_noise, noise)  # 计算损失
    
# -------------- DDPM 原始采样器 --------------
class SamplerDDPM:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion

        # 用到的属性，重新建立索引引用，不会额外占用显存
        self.q_posterior = diffusion.q_posterior
        self.betas = diffusion.betas
        self.sqrt_recip_alphas_cumprod = diffusion.sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = diffusion.sqrt_recipm1_alphas_cumprod
        self.predict_x0_from_eps = diffusion.predict_x0_from_eps

    def p_sample(self, model, x_t, t):
        """
        - model: 训练好的神经网络模型，用来预测噪声 ε。
        - x_t: 当前时间步 t 的图像张量。
        - t: 当前时间步的张量 (B,) 或 (1,)。

        Returns: x_{t-1} 的一个采样结果。
        """
        eps_theta = model(x_t, t)  # 模型预测噪声
        x_0_pred = self.predict_x0_from_eps(x_t, t, eps_theta)  # 预测 x_0
        mean, _, log_variance = self.q_posterior(x_0_pred, x_t, t)  # 后验均值方差

        if self.model_var_type == ModelVarType.FIXED_LARGE:
            pass
        elif self.model_var_type == ModelVarType.FIXED_SMALL:
            # 使用 beta_t（更小）作为方差 var
            beta_t = extract(self.betas, t, x_t.shape)
            log_variance = torch.log(beta_t.clamp(min=1e-20))
        else:
            raise NotImplementedError(f"Unknown model_var_type: {self.model_var_type}")

        # 从 N(mean, var) 中采样一个 x_{t-1}
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

    def p_sample_loop(self, model, shape, device):
        """
        从纯噪声开始反复调用 p_sample 采样出最终图像。

        - model: 已训练好的模型
        - shape: 要生成图像的 shape (例如 [batch, C, H, W])
        - device: 生成图像所在的设备

        Returns: x_0 的采样图像
        """
        x_t = torch.randn(shape, device=device)

        with torch.no_grad():
            for i in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), i, device=device, dtype=torch.long)
                x_t = self.p_sample(model, x_t, t)

        return x_t

    def sample(self, model, image_size, batch_size=16):
        """外部调用入口，封装 p_sample_loop"""
        shape = (batch_size, 3, image_size, image_size)
        device = next(model.parameters()).device  # 👈 自动获取模型所在设备
        return self.p_sample_loop(model, shape, device)
    
# -------------- DDIM 加速采样器 --------------
class SamplerDDIM:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
