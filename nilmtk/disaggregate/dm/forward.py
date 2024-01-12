import torch
import torch.nn as nn

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(beta_start, beta_end, timesteps):
    # beta_start = 0.0001
    # beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_beta_schedule(variant, beta_start, beta_end, timesteps):
    if variant == 'cosine':
        return cosine_beta_schedule(timesteps)
    elif variant == 'linear':
        return linear_beta_schedule(beta_start, beta_end, timesteps)
    elif variant == 'quadratic':
        return quadratic_beta_schedule(timesteps)
    elif variant == 'sigmoid':
        return sigmoid_beta_schedule(timesteps)
    else:
        raise NotImplemented


class GaussianForward(nn.Module):
    def __init__(self, num_timesteps=1000, schedule='linear', beta_start=1e-6, beta_end=0.006):
        super().__init__()

        self.num_timesteps = num_timesteps
        self.schedule = schedule

        # get process parameters
        self.register_buffer('betas', get_beta_schedule(self.schedule, beta_start, beta_end, self.num_timesteps))
        self.register_buffer('betas_sqrt', self.betas.sqrt())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, 0))
        self.register_buffer('alphas_cumprod_sqrt', self.alphas_cumprod.sqrt())
        self.register_buffer('alphas_cumprod_sqrt_prev',
                             torch.cat([torch.FloatTensor((1.0,)), self.alphas_cumprod_sqrt[:-1]]))
        self.register_buffer('alphas_one_minus_cumprod_sqrt', (1 - self.alphas_cumprod).sqrt())
        self.register_buffer('alphas_sqrt', self.alphas.sqrt())

    @torch.no_grad()
    def forward(self, x_0, t, return_noise=False, rand_level=False, predefined_noise=None):
        """
            Get noisy sample at t given x_0

            q(x_t | x_0)=N(x_t; alphas_cumprod_sqrt(t)*x_0, 1-alpha_cumprod(t)*I)
        """
        assert (t < self.num_timesteps).all()

        b = x_0.shape[0]
        noise_level = self.alphas_cumprod_sqrt[t]
        if rand_level:
            # uniformly sample between alpha_t and alpha_t-1
            rand = torch.rand_like(t, dtype=torch.float, device=t.device)
            noise_level = (noise_level - self.alphas_cumprod_sqrt_prev[t]) * rand + noise_level
        # print(noise_level)

        mean = x_0 * noise_level.view(b, 1, 1)  # 2d就是3个1
        std = self.alphas_one_minus_cumprod_sqrt[t].view(b, 1, 1)

        if predefined_noise is None:
            noise = torch.randn_like(x_0)  # todo: 这里改了
        else:
            noise = predefined_noise

        output = mean + std * noise

        if not return_noise:
            return output
        else:
            return output, noise, noise_level

    @torch.no_grad()
    def step(self, x_t, t, return_noise=False, predefined_noise=None):
        """
            Get next sample in the process

            q(x_t | x_t-1)=N(x_t; alphas_sqrt(t)*x_0,betas(t)*I)
        """
        assert (t < self.num_timesteps).all()

        mean = self.alphas_sqrt[t] * x_t
        std = self.betas_sqrt[t]

        if predefined_noise is None:
            noise = torch.randn_like(x_t)  # todo: 这里改了
        else:
            noise = predefined_noise

        output = mean + std * noise

        if not return_noise:
            return output
        else:
            return output, noise


if __name__ == '__main__':
    gf = GaussianForward()

    gf.forward(torch.Tensor([[[1,2,3]]]), torch.IntTensor([[900]]), rand_level=True)
