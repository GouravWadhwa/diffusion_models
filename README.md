# Improved Diffusion model

This repository is a working example of training and testing the diffusion models. This is based on the Improved diffusion model and also incorporates the details from the DDIM.

## Download the training dataset

For downloading the CIFAR10 dataset

```markdown
python dataset/cifar_dataset.py
```

## Training the models

For running the training of the diffusion model (Trains the model and stores the intemediatory results in ./logs/images/)

```markdown
python train.py --config_file_path configs/diffusion_v1.json
```

## Testing the models

For generating the samples from the diffusion model we can use the following code

```markdown
python generate_samples.py --config_file_path configs/diffusion_v1.json --checkpoint_path logs/model001000.pt --output_directory outputs/ --batch_size 4 --num_sample 4 --sample_timesteps 1000 --use_ddim False 
```

Here, sample_timesteps are the timesteps used for sampling from the model, use_ddim defines if you want to use the ddim algorithm for generating samples, num_samples defines the number of samples to generate in output_directory.
