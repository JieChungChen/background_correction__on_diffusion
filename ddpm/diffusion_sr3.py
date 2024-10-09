import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision


def extract(v, t):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view(-1, 1, 1, 1)


def make_beta_schedule(schedule, n_timestep, beta_0=1e-4, beta_T=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = torch.linspace(beta_0 ** 0.5, beta_T ** 0.5, n_timestep).double() ** 2
    elif schedule == 'linear':
        betas = torch.linspace(beta_0, beta_T, n_timestep).double() 
    elif schedule == 'const':
        betas = beta_T * torch.ones(n_timestep).double() 
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / torch.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * torch.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        # quadratic beta schedule
        self.register_buffer('betas', make_beta_schedule('quad', T, beta_1, beta_T))
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar).float())
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar).float())

    def forward(self, condit, x_0):
        t = torch.randint(0, self.T, size=(x_0.shape[0], ), device=x_0.device)
        sqrt_alphas_bar_t = torch.gather(self.sqrt_alphas_bar, 0, t).to(x_0.device)
        sqrt_one_minus_alphas_bar_t = torch.gather(self.sqrt_one_minus_alphas_bar, 0, t).to(x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (sqrt_alphas_bar_t.view(-1, 1, 1, 1) * x_0 + 
               sqrt_one_minus_alphas_bar_t.view(-1, 1, 1, 1) * noise)
        loss = F.mse_loss(self.model(torch.cat([condit, x_t], dim=1), t), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, beta_type, T):
        super().__init__()

        self.model = model
        self.T = T
        # quadratic beta schedule
        betas = make_beta_schedule(beta_type, T, beta_1, beta_T)
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat((torch.ones(1), alphas_bar[:-1]))

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('alphas_bar_prev', alphas_bar_prev)

        # for q(x_t | x_{t-1})
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / self.alphas_bar))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / self.alphas_bar - 1))
        # for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)
        self.register_buffer('posterior_var', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(np.maximum(posterior_variance, 1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_bar_prev) / (1. - alphas_bar))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_bar_prev) * np.sqrt(alphas) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return self.sqrt_recip_alphas_cumprod[t]* x_t - self.sqrt_recipm1_alphas_cumprod[t] * eps

    def q_posterior(self, x_t_prev, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_t_prev + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped
    
    def p_mean_variance(self, condit, x_t, t):
        # below: only log_variance is used in the KL computations
        eps = self.model(torch.cat([condit, x_t], dim=1), t)
        t = t[0]
        x_recon = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        x_recon.clamp_(-1., 1.)
        mean, log_var = self.q_posterior(x_t_prev=x_recon, x_t=x_t, t=t)
        return mean, log_var

    def forward(self, condit, x_T):
        x_t = x_T
        timesteps = (np.arange(self.T))[::-1].copy().astype(np.int64)
        for time_step in tqdm(timesteps, dynamic_ncols=True, desc='Denoise Step'):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(condit, x_t, t)
            noise = torch.randn_like(x_t) if time_step > 0 else torch.zeros_like(x_t)
            x_t = mean + noise * torch.exp(0.5 * var)
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            # if time_step%10 == 0:
            #     torchvision.utils.save_image(x_t, 'figures/%s.png'%str(time_step).zfill(4), normalize=True)
        x_0 = x_t
        return torch.clip(x_0, -1, 1) 