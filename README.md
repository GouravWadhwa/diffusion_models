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
