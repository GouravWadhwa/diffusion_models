import torch
import torch.distributed as dist

from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import os
import functools
import copy
import logging

import blobfile as bf
import numpy as np

from PIL import Image

from models.resample import UniformSampler, LossAwareSampler
from .dist_utils import (
    setup_dist, load_state_dict, sync_params, dev
)
from .stats_utils import RunningStats

class TrainLoop:
    def __init__(
        self, *, model, diffusion, data, batch_size, microbatch, lr, ema_rate, log_interval,
        save_interval, save_directory, log_image_interval, save_image_directory,
        resume_checkpoint, scheduler_sampler=None, weight_decay=0.0, lr_anneal_steps=0, log_file='training.log'
    ):
        super().__init__()

        self.log_file = log_file
        self._setup_logger(log_file=self.log_file, rank=dist.get_rank())

        self.model = model
        self.diffusion = diffusion
        self.data = data

        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size

        self.lr = lr
        self.lr_anneal_steps = lr_anneal_steps
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        )

        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_directory = save_directory
        self.save_image_directory = save_image_directory
        self.log_image_interval = log_image_interval
        
        self.resume_checkpoint = resume_checkpoint
        
        self.schedule_sampler = scheduler_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        
        self.sync_cuda = torch.cuda.is_available()

        self._load_and_sync_parameters()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.statistics = RunningStats()
        
        if self.resume_checkpoint:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model, device_ids=[dev()], output_device=dev(), broadcast_buffers=False, bucket_cap_mb=128, find_unused_parameters=False
            )
        else:
            if dist.get_world_size() > 1:
                self.logger.warn("Distributed training requires CUDA, Gradients might not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model


    def _setup_logger(self, log_file, rank):
        self.logger = logging.getLogger("TrainLoop")
        self.logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)

        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)

        c_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
        f_handler.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)

        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        if not self.logger.handlers:
            self.logger.addHandler(c_handler)
            self.logger.addHandler(f_handler)


    def _load_and_sync_parameters(self):
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)

            if dist.get_rank() == 0:
                self.logger.info(f"loading model from checkpoint: {self.resume_checkpoint} ....")
                self.model.load_state_dict(
                    load_state_dict(self.resume_checkpoint, map_locatio=dev())
                )
        
        sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        opt_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            self.logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint} ....")

            state_dict = load_state_dict(
                opt_checkpoint, map_location=dev()
            )

            self.optimizer.load_state_dict(state_dict)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        ema_checkpoint = bf.join(
            bf.dirname(self.resume_checkpoint), f"ema_{rate}_{self.resume_step:06}.pt"
        )
        if bf.exists(ema_checkpoint):
            if dist.get_rank() == 0:
                self.logger.info(f"loading EMA from the checkpoint: {ema_checkpoint} ....")
                state_dict = load_state_dict(
                    ema_checkpoint, map_location=dev()
                )

                ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]

            sync_params(ema_params)
            return ema_params
        
    def run_loop(self):
        while(
            not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            if self.step % self.log_interval == 0:
                self.log_step()

            if self.step % self.save_interval == 0:
                self.save()
            
            self.step += 1
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()


    def log_step(self):
        self.logger.info(
            f"[Step: {self.step}] Samples processed: {(self.step + self.resume_step + 1) * self.global_batch} || " + " || ".join(
                [f"{param}: {self.statistics.get_averge(param)[0]}" for param in self.statistics.params]
            )
        )
    
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimizer_normal()
    
    def optimizer_normal(self):
        self._log_grad_norm()
        self._anneal_lr()

        self.optimizer.step()

        for rate, params in zip(self.ema_rate, self.ema_params):
            self.update_ema(params, self.model.parameters(), rate=rate)

    def update_ema(self, target_params, source_params, rate=0.99):
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(rate).add_(src, alpha=1-rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for name, p in list(self.model.named_parameters()):
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
            else:
                print(name)
        self.statistics.step("grad_norm", np.sqrt(sqsum), 1)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def forward_backward(self, batch, cond):
        zero_grad(self.model.parameters())

        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dev())

            micro_cond = {
                k: v[i : i + self.microbatch].to(dev()) for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses, self.ddp_model, micro, t, model_kwargs=micro_cond
            )

            if last_batch or not self.use_ddp:
                losses, pred_x0, x_t = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses, pred_x0, x_t = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()

            # Adding the logs
            for key, value in losses.items():
                self.statistics.step(key, value.mean(), 1)

                for sub_t, sub_loss in zip(t.cpu().numpy(), value.detach().cpu().numpy()):
                    quartile = int(4 * sub_t / self.diffusion.num_timesteps)
                    self.statistics.step(f"{key}_q{quartile}", sub_loss, 1)

            loss.backward()

            if self.step % self.log_image_interval == 0:
                self.logger.info("Saving training images ....")
                self.save_images(micro, t, micro_cond, pred_x0, x_t)

    def save_images(self, input_images, timesteps, class_names, pred_x0, x_t):
        input_images = input_images.detach().cpu().numpy()
        timesteps = timesteps.detach().cpu().numpy()
        pred_x0 = pred_x0.detach().cpu().numpy()
        x_t = x_t.detach().cpu().numpy()

        if "y" not in class_names:
            class_names["y"] = [-1] * input_images.shape[0]
        
        for i, (original_image, t, class_category, prediction, input_image) in enumerate(zip(input_images, timesteps, class_names["y"], pred_x0, x_t)):
            input_image = np.transpose((input_image + 1) * 127.5, (1, 2, 0))
            original_image = np.transpose((original_image + 1) * 127.5, (1, 2, 0))
            prediction = np.transpose((np.clip(prediction, -1, 1) + 1) * 127.5, (1, 2, 0))

            output_image = np.concatenate([original_image, input_image, prediction], axis=1).astype(np.uint8)
            output_image = Image.fromarray(output_image)

            filename = f"image_{self.step}_{i}_{t}_{class_category}_.jpg"
            output_image.save(bf.join(self.save_image_directory, filename), "JPEG")


    def save(self):
        def save_checkpoint(rate, state_dict):
            if dist.get_rank() == 0:
                self.logger.info(f"saving model {rate} ....")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                
                with bf.BlobFile(bf.join(self.save_directory, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, self._params_to_state_dict(params))

        if dist.get_rank() == 0:
            filename = f"opt{(self.step + self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(self.save_directory, filename), "wb") as f:
                torch.save(self.optimizer.state_dict(), f)

        dist.barrier()

    def _params_to_state_dict(self, model_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict

            state_dict[name] = model_params[i]
        return state_dict


def zero_grad(model_params):
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

def parse_resume_step_from_filename(filename):
    """
    Parse filename of the form /path/to/modelNNNN.pt where NNNN is the checkpoint's number of steps
    """

    split = filename.split("model")
    if len(split) < 2:
        return 0
    try:
        return int(split[-1].split(".")[0])
    except:
        return 0
        