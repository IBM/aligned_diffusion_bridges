import math
from functools import partial

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Timesteps ----------------------

def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py"""
    # This is inspired by section 3.5 in Attention is all you need.
    if len(timesteps.shape) > 1:
        timesteps = timesteps.squeeze(-1)
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return emb


def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=1000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim, max_positions=1000))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func


def get_t_schedule(inference_steps, t_max=1.0):
    return np.linspace(0, t_max, inference_steps + 1)


def beta(g, ts, steps_num):
    dt = 1/steps_num

    if isinstance(ts, torch.Tensor):
        ts = ts.cpu().numpy()
    elif isinstance(ts, int):
        ts = [ts]
        
    beta_t = torch.zeros((len(ts), 1))

    for i, t in enumerate(ts):
        ks = np.arange(0, np.floor(t * steps_num))
        beta_t[i] = np.square(g(ks/steps_num)).sum() * dt

    return beta_t



def sample_from_brownian_bridge(g, t, x_0, x_T, t_min=0.0, t_max=1.0):
    # Taken from https://en.wikipedia.org/wiki/Brownian_bridge#General_case
    assert x_0.shape == x_T.shape, "End points of Brownian bridge are not of same shape"
    assert t_max > t_min, "Start time is larger than end time"
    if isinstance(t, float):
        t = torch.tensor(t).expand(x_0.shape)

    if t.shape != x_0.shape:
        exp_t = t.expand(x_0.shape)
    else:
        exp_t = t

    mu_t = x_0 + ( (exp_t - t_min) * (x_T - x_0) / (t_max - t_min) )

    # TODO: Check the exact formula (absence of sqrt?)
    sigma_t = torch.sqrt((t_max - exp_t) * (exp_t - t_min) / (t_max - t_min)) * g(t)
    return mu_t + sigma_t * torch.randn_like(exp_t)


# --------- Diffusivity Schedule ---------

def constant_g(t, g_max):
    return np.ones_like(t) * g_max

def triangular_g(t, g_max):
    g_min = 0.85
    return g_max - 2 * np.abs(t - .5) * (g_max-g_min)

def inverse_triangular_g(t, g_max):
    g_min = .01
    return g_min - 2 * np.abs(t - .5) * (g_min-g_max)

def decreasing_g(t, g_max):
    g_min = .1
    return g_max - np.square(t) * (g_max-g_min)

diffusivity_schedules = {
    "constant": constant_g,
    "triangular": triangular_g,
    "inverse_triangular": inverse_triangular_g,
    "decreasing": decreasing_g,
}

def get_diffusivity_schedule(schedule, g_max):
    return partial(diffusivity_schedules[schedule], g_max=g_max)
