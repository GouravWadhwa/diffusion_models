from abc import ABC, abstractmethod

import enum

import torch
import torch.distributed as dist
import numpy as np

class Sampler(enum.Enum):
    UNIFORM = "uniform"
    LOSS_SECOND_MOMENT = "loss_second_moment"

def create_sampler(sampler, diffusion):
    if sampler is Sampler.UNIFORM:
        return UniformSampler(diffusion=diffusion)
    elif sampler is Sampler.LOSS_SECOND_MOMENT:
        return LossSecondMomentumResampler(diffsion=diffusion)
    else:
        raise NotImplementedError(f"Unknow schedule sampler: {sampler}")

class ScheduledSampler(ABC):
    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step
        The weights must be positive
        """
    
    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)

        indices_np = np.random.choice(len(p), size=(batch_size, ), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)

        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)

        return indices, weights
    
class UniformSampler(ScheduledSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights
    
class LossAwareSampler(ScheduledSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device) for _ in range(dist.get_world_size())
        ]

        dist.all_gather(batch_sizes, torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device))

        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]

        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from the model
        """

class LossSecondMomentumResampler(LossAwareSampler):
    def __init__(self, diffsion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffsion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob

        self._loss_history = np.zeros(
            [diffsion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffsion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)

        weights *= (1 - self.uniform_prob)
        weights += self.uniform_prob / len(weights)

        return weights
    
    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()