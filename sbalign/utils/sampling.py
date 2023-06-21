import torch
import numpy as np

from sbalign.utils.definitions import DEVICE


def sampling(pos_0, model, g, inference_steps, t_schedule, apply_score=False, return_traj: bool=False):
    pos = pos_0.clone()
    model.eval()

    trajectory = np.zeros((inference_steps+1, *pos_0.shape))
    trajectory[0] = pos.cpu()

    dt = t_schedule[1] - t_schedule[0]

    with torch.no_grad():
        for t_idx in range(1, inference_steps+1):
            t = t_schedule[t_idx]

            drift_pos = model.run_drift(pos, torch.ones(pos.shape[0]).to(DEVICE)* t)

            if apply_score:
                assert False, "Must pass x_T as parameter of the function"
                # torch.stack([torch.ones(pos.shape[0], device=DEVICE)*5, pos[:,1]], axis=1)
                # drift_pos = drift_pos + model.run_doobs_score(pos, ..., torch.ones(pos.shape[0]).to(DEVICE)* t)

            diffusion = g(t) * torch.randn_like(pos) * torch.sqrt(dt)

            dpos = np.square(g(t)) * drift_pos * dt + diffusion
            pos = pos + dpos

            trajectory[t_idx] = pos.cpu()

    if return_traj:
        return trajectory
    else:
        return trajectory[-1]

