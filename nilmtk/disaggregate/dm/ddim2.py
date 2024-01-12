"""

    This file contains the DDIM sampler class for a diffusion process

"""
import torch
from torch import nn

# from forward import get_beta_schedule
# from samplers.fixed_ddim import make_ddim_schedule


def make_ddim_schedule(num_infer_steps):
    if num_infer_steps == 8:
        return torch.Tensor([1e-6, 2e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 9e-1])
    raise RuntimeError("Does not support time steps of " + num_infer_steps)


class DDIM_Sampler2(nn.Module):

    def __init__(self,
                 model,
                 num_timesteps=8,
                 clip_sample=True,
                 schedule=None
                 ):

        super().__init__()

        self.model = model
        self.num_timesteps = num_timesteps
        # self.ratio = self.train_timesteps // self.num_timesteps
        self.final_alpha_cumprod = torch.tensor([1.0])
        self.clip_sample = clip_sample
        self.schedule = schedule

        if schedule is None:
            schedule = make_ddim_schedule(self.num_timesteps)

        # self.register_buffer('betas', get_beta_schedule(self.schedule, self.train_timesteps), False)
        self.register_buffer('betas', schedule, False)
        self.register_buffer('alphas', 1 - self.betas, False)
        self.register_buffer('alphas_cumprod', self.alphas.cumprod(dim=0),
                             False)
        self.register_buffer(
            'alphas_cumprod_prev',
            torch.cat([torch.FloatTensor([1.]), self.alphas_cumprod[:-1]]),
            False)
        alphas_cumprod_prev_with_last = torch.cat(
            [torch.FloatTensor([1.]), self.alphas_cumprod])
        self.register_buffer('sqrt_alphas_cumprod_prev',
                             alphas_cumprod_prev_with_last.sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod', self.alphas_cumprod.sqrt(),
                             False)
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             (1. / self.alphas_cumprod).sqrt(), False)
        self.register_buffer('sqrt_alphas_cumprod_m1',
                             (1. - self.alphas_cumprod).sqrt() *
                             self.sqrt_recip_alphas_cumprod, False)
        posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) \
                             / (1 - self.alphas_cumprod)
        posterior_variance = torch.stack(
            [posterior_variance,
             torch.FloatTensor([1e-20] * self.num_timesteps)])
        posterior_log_variance_clipped = posterior_variance.max(
            dim=0).values.log()
        posterior_mean_coef1 = self.betas * self.alphas_cumprod_prev.sqrt() / (
                1 - self.alphas_cumprod)
        posterior_mean_coef2 = (1 - self.alphas_cumprod_prev
                                ) * self.alphas.sqrt() / (1 -
                                                          self.alphas_cumprod)
        self.register_buffer('posterior_log_variance_clipped',
                             posterior_log_variance_clipped, False)
        self.register_buffer('posterior_mean_coef1',
                             posterior_mean_coef1, False)
        self.register_buffer('posterior_mean_coef2',
                             posterior_mean_coef2, False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def q_sample(self, y_0, step=None, noise_level=None, eps=None):
        batch_size = y_0.shape[0]
        if noise_level is not None:
            continuous_sqrt_alpha_cumprod = noise_level
        elif step is not None:
            continuous_sqrt_alpha_cumprod = self.sqrt_alphas_cumprod_prev[step]
        assert (step is not None or noise_level is not None)
        if isinstance(eps, type(None)):
            eps = torch.randn_like(y_0, device=y_0.device)
        outputs = continuous_sqrt_alpha_cumprod * y_0 + (
                1. - continuous_sqrt_alpha_cumprod ** 2).sqrt() * eps
        return outputs

    def q_posterior(self, y_0, y, step):
        posterior_mean = self.posterior_mean_coef1[step] * y_0 \
                         + self.posterior_mean_coef2[step] * y
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[step]
        return posterior_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def predict_start_from_noise(self, y, t, eps):
        return self.sqrt_recip_alphas_cumprod[t].unsqueeze(
            -1) * y - self.sqrt_alphas_cumprod_m1[t].unsqueeze(-1) * eps

    # t: interger not tensor
    @torch.no_grad()
    def p_mean_variance(self, y, y_down, t, time, clip_denoised: bool):
        batch_size = y.shape[0]
        noise_level = self.sqrt_alphas_cumprod_prev[t + 1].repeat(
            batch_size, 1)
        eps_recon = self.model(y, y_down, t, noise_level=noise_level, time=time)  # todo: 这里的t不对 好像对的
        y_recon = self.predict_start_from_noise(y, t, eps_recon)
        if clip_denoised:
            y_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance_clipped = self.q_posterior(
            y_recon, y, t)
        return model_mean, posterior_log_variance_clipped

    @torch.no_grad()
    def compute_inverse_dynamincs(self, y, y_down, t, time, clip_denoised=False):
        model_mean, model_log_variance = self.p_mean_variance(
            y, y_down, t, time, clip_denoised)
        eps = torch.randn_like(y) if t > 0 else torch.zeros_like(y)
        return model_mean + eps * (0.5 * model_log_variance).exp()
