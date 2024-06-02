import torch

from .gaussian_diffusion import GaussianDiffusion

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
                
            raise ValueError(f"Cannot create exactly {num_timesteps} steps with an integer stride")
        
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)

    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"Cannot divide section of {size} steps into {section_counts}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size

    return set(all_steps)

class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip timesteps in the base diffusion process

    :params
    use_timesteps: a collection of timesteps from the original diffusion to be used.
    kwargs: the params used for the base diffusion model.
    """

    def __init__(self, use_tiemsteps, **kwargs):
        self.use_timesteps = set(use_tiemsteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0

        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod

                self.timestep_map.append(i)

        kwargs["betas"] = torch.tensor(new_betas)
        super().__init__(**kwargs)
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        return super().p_mean_variance(self._wrapped_model(model), x, t, clip_denoised, denoised_fn, model_kwargs)
    
    def training_losses(self, model, x0, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise=None):
        return super().training_losses(self._wrapped_model(model), x0, t, clip_denoised, denoised_fn, model_kwargs, noise)
    
    def _wrapped_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        return t

class _WrappedModel:
    def __init__(self, model, timestep_map, rescal_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescal_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        
        return self.model(x, new_ts, **kwargs)