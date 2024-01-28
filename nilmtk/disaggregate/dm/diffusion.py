import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from nilmtk.disaggregate.dm.forward import *
from nilmtk.disaggregate.dm.ddpm import DDPM_Sampler
from nilmtk.disaggregate.dm.ddim2 import DDIM_Sampler2


class ConditionalDiffusion(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 generated_channels=3,
                 condition_channels=3,
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 beta_start=1e-6,
                 beta_end=0.006,
                 num_timesteps=1000,
                 sampler=None,
                 apply_input_t=True
                 ):
        super().__init__()

        # Basic Params
        self.generated_channels = generated_channels
        self.condition_channels = condition_channels
        self.num_timesteps = num_timesteps
        self.loss_fn = loss_fn

        self.rand_level = True
        self.apply_input_t = apply_input_t

        # Forward Process
        self.forward_process = GaussianForward(num_timesteps=self.num_timesteps,
                                               schedule=schedule,
                                               beta_start=beta_start,
                                               beta_end=beta_end)

        # Neural Network Backbone
        self.model = model

        # defaults to a DDPM sampler if None is provided
        self.sampler = DDPM_Sampler(
            num_timesteps=self.num_timesteps) if sampler is None else sampler

    # def set_guide_w(self,
    #                 guide_w):
    #     self.guide_w = guide_w

    def input_T(self, input_):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input_.clip(0, 1).mul_(2)).sub_(1) if self.apply_input_t else input_
        # return input_

    def output_T(self, input_):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input_.add_(1)).div_(2) if self.apply_input_t else input_
        # return input_

    @torch.no_grad()
    def forward(self,
                condition,
                sampler=None,
                verbose=False
                ):
        """
            forward() function triggers a complete inference cycle
            
            A custom sampler can be provided as an argument!
        """
        condition = self.input_T(condition)

        # read dimensions
        # b,c,h,w=condition.shape  # for 2d
        # print("Forward shape", condition.shape)
        b, c, length = condition.shape  # for 1d
        device = next(self.model.parameters()).device
        # condition=condition.repeat(2)
        condition = condition.to(device)

        # select sampler
        if sampler is None:
            sampler = self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps
        it = reversed(range(0, num_timesteps))

        if isinstance(sampler, DDIM_Sampler2):
            init_noise = True
            batch_size = b
            start_step = num_timesteps
            step = torch.tensor([start_step] * batch_size,
                                dtype=torch.long,
                                device=device)
            # y_t = torch.randn_like(
            #     condition, device=device) if init_noise \
            #     else sampler.q_sample(condition, step=step)
            y_t = torch.randn([b, self.generated_channels, length], device=device) if init_noise \
                else sampler.q_sample(condition, step=step)
            ys = [y_t]
            # t = start_step - 1
            for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                y_t = sampler.compute_inverse_dynamincs(y_t, condition, i, t)
                ys.append(y_t)

            return self.output_T(ys[-1])

        else:
            x_t = torch.randn([b, self.generated_channels, length], device=device)
            # x_t_2 = x_t.repeat((2, 1, 1))  # weight

            for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
                t = torch.full((b,), i, device=device, dtype=torch.long)
                noise_level = self.forward_process.alphas_cumprod_sqrt[t]
                z_t = self.model(x_t, condition, t=i, noise_level=noise_level.unsqueeze(-1),
                                 time=t)  # prediction of noise
                x_t = sampler(x_t, t, z_t)  # prediction of next state

            return self.output_T(x_t)

    def train_step(self, output, condition):
        output = self.input_T(output)
        condition = self.input_T(condition)

        b, c, length = output.shape  # for 1d
        device = output.device

        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise, noise_level = self.forward_process(
            output, t, return_noise=True, rand_level=self.rand_level)

        # print(output_noisy.shape, condition.shape)

        # reverse pass
        noise_hat = self.model(output_noisy, condition, t=None,
                               noise_level=noise_level.unsqueeze(-1), time=t)

        return noise, noise_hat

    def freeze(self, freeze=True):
        self.model.freeze(freeze)

    # def p_loss(self, output, condition, return_pred=False):
    #     """
    #         Assumes output and input are in [-1,+1] range
    #     """
    #     output = self.input_T(output)
    #     condition = self.input_T(condition)
    #
    #     # b,c,h,w=output.shape  # for 2d
    #     b, c, length = output.shape  # for 1d
    #     device = output.device
    #
    #     # loss for training
    #
    #     # input is the optional condition
    #     t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
    #     output_noisy, noise, noise_level = self.forward_process(
    #         output, t, return_noise=True, rand_level=self.rand_level)
    #
    #     # print(output_noisy.shape, condition.shape)
    #
    #     # reverse pass
    #     # model_input = torch.cat([output_noisy, condition], 1).to(device)
    #     noise_hat = self.model(output_noisy, condition, t=None,
    #                            noise_level=noise_level.unsqueeze(-1), time=t)
    #
    #     # apply loss
    #     loss = self.loss_fn(noise, noise_hat)
    #     if return_pred:
    #         return loss, t, output - noise_hat
    #     else:
    #         return loss
