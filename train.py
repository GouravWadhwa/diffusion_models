import argparse

import torch
import torch.distributed as dist

import os
import json
import logging

from dataset.image_datasets import load_data

from training.dist_utils import setup_dist
from training.train_utils import TrainLoop

from models.unet import UNet
from models.spaced_diffusion import SpacedDiffusion, space_timesteps
from models.gaussian_diffusion import ModelMeanType, ModelVarianceType, LossType, make_beta_schedule
from models.resample import create_sampler, Sampler

def main():
    parser = argparse.ArgumentParser(description="training the Improved Diffusion model")

    parser.add_argument("--config_file_path", type=str, required=True, help="the config file used for the model")
    parser.add_argument("--resume_checkpoint", type=str, required=False, default="", help="the checkpoint path from which to resume training")
    args = parser.parse_args()

    print(f"Loading the params from the config file: {args.config_file_path} ....")
    with open(args.config_file_path, "r") as f:
        params = json.loads(f.read())

    print("creating the dataloader ....")
    dataloader = load_data(params["data_params"])

    print("Setting up the ditributed training ....")
    setup_dist()

    print("Creating the model ....")
    if ModelVarianceType(params["diffusion_params"]["model_var_type"]) in [ModelVarianceType.LEARNED, ModelVarianceType.LEARNED_RANGE]:
        params["model_params"]["output_channels"] *= 2
    model = UNet(params["model_params"])

    print("Creating the gaussian diffusion ....")
    timesteps_respacing = params["diffusion_params"]["timesteps_respacing"]
    if not timesteps_respacing:
        timesteps_respacing = [params["diffusion_params"]["diffusion_steps"]]
    
    diffusion = SpacedDiffusion(
        use_tiemsteps=space_timesteps(params["diffusion_params"]["diffusion_steps"], timesteps_respacing),
        betas=make_beta_schedule(params["diffusion_params"]["betas_schedule"], n_timesteps=params["diffusion_params"]["diffusion_steps"]),
        model_mean_type=ModelMeanType(params["diffusion_params"]["model_mean_type"]),
        model_var_type=ModelVarianceType(params["diffusion_params"]["model_var_type"]),
        model_loss_type=LossType(params["diffusion_params"]["loss_type"]),
        rescale_timesteps=params["diffusion_params"]["rescale_timesteps"]
    )

    print("Creating the save directory and log file ....")
    os.makedirs(params["trainer_params"]["save_directory"], exist_ok=True)
    with open(params["trainer_params"]["log_file"], "w"):
        pass


    print("Starting the training ....")
    TrainLoop(
        model=model, diffusion=diffusion, data=dataloader,
        batch_size=params["data_params"]["batch_size"],
        microbatch=params["data_params"]["microbatch_size"],
        lr=params["trainer_params"]["lr"],
        ema_rate=params["trainer_params"]["ema_rate"],
        log_interval=params["trainer_params"]["log_interval"],
        save_interval=params["trainer_params"]["save_interval"],
        log_image_interval=params["trainer_params"]["log_image_interval"],
        save_image_directory=params["trainer_params"]["save_image_directory"],
        save_directory=params["trainer_params"]["save_directory"],
        resume_checkpoint=args.resume_checkpoint,
        scheduler_sampler=create_sampler(Sampler(params["diffusion_params"]["noise_schedule"]), diffusion=diffusion),
        weight_decay=params["trainer_params"]["weight_decay"],
        lr_anneal_steps=params["trainer_params"]["lr_anneal_steps"],
        log_file=params["trainer_params"]["log_file"]
    ).run_loop()


if __name__ == "__main__":
    main()