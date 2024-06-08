import argparse

import torch
import torch.distributed as dist

import os
import json

import matplotlib.pyplot as plt

from training.dist_utils import setup_dist, dev

from models.gaussian_diffusion import ModelMeanType, ModelVarianceType, make_beta_schedule, LossType
from models.spaced_diffusion import SpacedDiffusion, space_timesteps
from models.unet import UNet

def main():
    parser = argparse.ArgumentParser(description="testing the improved diffusion model")

    parser.add_argument("--config_file_path", type=str, required=True, help="config file used for the model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="the checkpoint path for which we need to evaluate")
    parser.add_argument("--sample_timesteps", type=str, required=False, default=None, help="total timesteps to respace for gaussian diffusion")
    parser.add_argument("--output_directory", type=str, required=True, help="path to store the outputs")
    parser.add_argument("--num_samples", type=int, required=False, default=50, help="number of images generated using diffusion")
    parser.add_argument("--batch_size", type=int, required=False, default=32, help="batch size for the generating samples")
    parser.add_argument("--use_ddim", type=bool, required=False, default=False, help="if true then will use the ddim algorithm")
    args = parser.parse_args()

    print(f"Loading the params from the config file: {args.config_file_path} ....")
    with open(args.config_file_path, "r") as f:
        params = json.loads(f.read())

    print("Setting up the ditributed training ....")
    setup_dist()

    print("Creating the model ....")
    if ModelVarianceType(params["diffusion_params"]["model_var_type"]) in [ModelVarianceType.LEARNED, ModelVarianceType.LEARNED_RANGE]:
        params["model_params"]["output_channels"] *= 2
    model = UNet(params["model_params"])
    model.to(dev())

    print("Creating the gaussian diffusion ....")
    timesteps_respacing = args.sample_timesteps
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

    print("Creating the save directory ....")
    os.makedirs(args.output_directory, exist_ok=True)

    print("sampling ....")
    all_images = []
    all_labels = []

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if params["diffusion_params"]["class_cond"]:
            classes = torch.randint(low=0, high=params["model_params"]["num_classes"], size=(args.batch_size, ), device=dev())
            model_kwargs["y"] = classes

        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

        sample = sample_fn(model, (args.batch_size, 3, params["data_params"]["resolution"], params["data_params"]["resolution"]), model_kwargs=model_kwargs, progress=True)
        sample = ((sample[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)

        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        if params["diffusion_params"]["class_cond"]:
            gathered_classes = [torch.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_classes, classes)

            all_labels.extend([labels.cpu().numpy() for labels in gathered_classes])

    if dist.get_rank() == 0:
        for i, (image, label) in enumerate(zip(all_images, all_labels)):
            for j in range(image.shape[0]):
                plt.imsave(os.path.join(args.output_directory, f"image_{i*image.shape[0] + j}_{label[j]}.jpg"), image[j, ...])

if __name__ == "__main__":
    main()