import math
import enum

import torch
import torch.nn as nn

from .losses import kl_divergence, dicretixed_gaussian_log_likelihood

def make_beta_schedule(schedule, n_timesteps, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "quad":
        betas = torch.linspace(
            start=linear_start ** 0.5, end=linear_end ** 0.5, steps=n_timesteps, dtype=torch.float64
        ) ** 2
    elif schedule == "linear":
        betas = torch.linspace(
            start=linear_start, end=linear_end, steps=n_timesteps, dtype=torch.float64
        )
    elif schedule == "cosine":
        timesteps = (
            torch.arange(0, n_timesteps + 1, dtype=torch.float64) / n_timesteps + cosine_s
        )

        alphas = (timesteps / (1 + cosine_s)) * (math.pi / 2)
        alphas = torch.cos(alphas).pow(2)

        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])

        betas = betas.clamp(max = 0.999)

    return betas

def _extract_into_tensor(value_arr, timesteps, shape):
    output = torch.gather(value_arr, 0, timesteps)
    reshape = [shape[0]] + [1 for _ in range(len(shape) - 1)]

    output = output.reshape(*reshape)

    return output.expand(shape)

class ModelMeanType(enum.Enum):
    PREVIOUS_X = "previous_x"
    START_X = "start_x"
    EPSILON = "epsilon"

class ModelVarianceType(enum.Enum):
    LEARNED = "learned"
    FIXED_SMALL = "fixed_small"
    FIXED_LARGE = "fixed_large"
    LEARNED_RANGE = "learned_range"

class LossType(enum.Enum):
    # uses raw MSE loss (and KL when learning variance)
    MSE = "mse"

    # use raw MSE loss (with RESCALED_KL when learning variances)
    RESCALED_MSE = "rescaled_mse"

    # use the variational lower bound
    KL = "kl"

    # similar to KL, but rescale to estimate the full VLB
    RESCALED_KL = "rescaled_kl"


class GaussianDiffusion(nn.Module):
    def __init__(self, betas, model_mean_type, model_var_type, model_loss_type, rescale_timesteps=False):
        super().__init__()

        betas = betas.type(torch.float64)
        self.num_timesteps = int(betas.shape[0])

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.model_loss_type = model_loss_type
        self.rescale_timesteps = rescale_timesteps

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), dim=0
        )
        alphas_cumprod_next = torch.cat([alphas_cumprod[1:], torch.tensor([0], dtype=torch.float64)], dim=0)

        posterior_variance = betas * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod))

        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)
        posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register("alphas_cumprod_next", alphas_cumprod_next)

        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register("posterior_mean_coef1", posterior_mean_coef1)
        self.register("posterior_mean_coef2", posterior_mean_coef2)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recip_minus_one_alphas_cumprod", torch.sqrt((1 / alphas_cumprod) - 1))


    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x_start, t, noise=None):
        """
        diffuse the data for the given number of steps

        :params
        x_start: starting image
        t: the number of diffusion timesteps (t=0 means 1 timestep)
        noise: noise to be added, if not given is from normal distribution

        :returns
        A sample from q(x_t | x_0)
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        output = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return output

    def q_mean_variace(self, x_start, t):
        """
        Get the distribution q(x_t | x_0)

        :params
        x_start: starting image
        t: the number of diffusion timesteps (t=0 means 1 timestep)
        
        :returns
        a dict with following keys
        mean, variance, log_variance
        """

        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)

        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance
        }
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Get the posterior distribution q(x_{t-1} | x_t, x_0)

        :params
        x_start: starting image
        x_t: image after t timesteps of diffusion
        t: number of timesteps

        :retuens
        a dict with following keys
        mean, variance, log_variance
        """

        mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start + _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) * x_t
        variance = _extract_into_tensor(self.posterior_variance, t, x_start.shape)
        log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_start.shape)

        return {
            "mean": mean,
            "variance": variance,
            "log_variance": log_variance
        }
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
    
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Appy the model to get p(x_{t-1} | x_t), as well as the prediction x_0

        :params
        model: the model which takes the signal and a batch of timesteps as input
        x: the signal for the timestep t
        t: the value of t, starting at 0 for the first diffusion step
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.

        :returns
        a dict with following keys
        mean, variance, log_variance, x_0
        """

        def process_outputs(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            
            if clip_denoised:
                x = x.clamp(-1, 1)

            return x

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarianceType.LEARNED, ModelVarianceType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:]), "[Gaussian Diffusion] While predicting variance, the number of channels should be double"

            model_output, model_var_output = torch.split(model_output, C, dim=1)
            if self.model_var_type is ModelVarianceType.LEARNED:
                model_log_variance = model_var_output
                model_variance = torch.exp(model_var_output)
            elif self.model_var_type is ModelVarianceType.LEARNED_RANGE:
                max_log = _extract_into_tensor(torch.log(self.betas), t, x.shape)
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

                v = (model_var_output + 1) / 2
                model_log_variance = v * max_log + (1 - v) * min_log
                model_variance = torch.exp(model_log_variance)
        elif self.model_var_type in [ModelVarianceType.FIXED_LARGE, ModelVarianceType.FIXED_SMALL]:
            if self.model_var_type is ModelVarianceType.FIXED_LARGE:
                model_log_variance = torch.log(self.betas)
                model_variance = self.betas
            elif self.model_var_type is ModelVarianceType.FIXED_SMALL:
                model_log_variance = torch.log(self.posterior_variance)
                model_variance = self.posterior_variance

            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            model_variance = _extract_into_tensor(model_variance, t, x.shape)

        if self.model_mean_type is ModelMeanType.EPSILON:
            x0 = process_outputs(
                self._get_x0_from_epsilon(x_t=x, t=t, eps=model_output)
            )
            q_posterior = self.q_posterior_mean_variance(x0, x, t)
            model_mean = q_posterior["mean"]
        elif self.model_mean_type is ModelMeanType.PREVIOUS_X:
            x0 = process_outputs(
                self._get_x0_from_xprev(x_t=x, t=t, x_prev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type is ModelMeanType.START_X:
            x0 = process_outputs(model_output)
            q_posterior = self.q_posterior_mean_variance(x0, x, t)
            model_mean = q_posterior["mean"]

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_x0": x0
        }

    def _get_x0_from_epsilon(self, x_t, t, eps):
        assert x_t.shape == eps.shape, "[Gaussian Diffusion | _get_x0_from_epsilon] shape of prediction and input should be same"

        x0 = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(self.sqrt_recip_minus_one_alphas_cumprod, t, x_t.shape) * eps
        return x0
    
    def _get_x0_from_xprev(self, x_t, t, x_prev):
        assert x_t.shape == x_prev.shape, "[Gaussian Diffusion | _get_x0_from_xprev] shape of prediction and input should be same"

        x0 = _extract_into_tensor(1 / self.posterior_mean_coef1, t, x_t.shape) * x_prev - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t) * x_t
        return x0
    
    def _get_eps_from_x0(self, x_t, t, pred_x0):
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_x0) / _extract_into_tensor(self.sqrt_recip_minus_one_alphas_cumprod, t, x_t.shape)
        return eps
    
    def p_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Sample x_{t-1} from the model given the timestep

        :parmas
        model: the model for predicting the p(x_{t-1} | x_{t})
        x: the current tensor at x_{t}
        t: the value of t, starting at 0 for the first diffusion step
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.

        :returns
        a dict with the following keys
        sample: a random sampel from the model
        pred_x0: a prediction of x0
        """

        out = self.p_mean_variance(
            model=model, x=x, t=t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {
            "sample": sample,
            "pred_x0": out["pred_x0"]
        }
    
    def p_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False):
        """
        Generates the samples from the model

        :params
        model: the model for predicting p(x_{t-1} | x_{t})
        shape: the shape of the image to be predicted
        noise: if specified the output of the revious timestep
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        device: if specified the device used for creting sample on. If not then uses the model device
        progress: if True, shows a tqdm progress bar

        :returns
        a dict with the samples from all the timesteps.
        """

        if device is None:
            deice = next(model.parameters()).device

        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        samples = {}
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
                )
                img = out["sample"]
                samples[i] = img

        return samples
        

    def ddim_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM

        :params
        model: the model which takes the signal and a batch of timesteps as input
        x: the signal for the timestep t
        t: the value of t, starting at 0 for the first diffusion step
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        eta: the muliplication factor of the posterior distriution

        :returns
        a dict with following keys
        sample: a sample from the model
        pred_x0: a predition of x0
        """

        out = self.p_mean_variance(
            model, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )

        # In case the model predicts x_start and x_prev
        eps = self._get_eps_from_x0(x, t, out["pred_x0"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        noise = torch.randn_like(x)
        mean_pred = out["pred_x0"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

        nonzer_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = mean_pred + nonzer_mask * noise * sigma
        return {
            "sample": sample,
            "pred_x0": out["pred_x0"]
        }
    
    def ddim_reverse_sample(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0):
        """
        Sample x_{t+1} from the model using DDIM reverse

         :params
        model: the model which takes the signal and a batch of timesteps as input
        x: the signal for the timestep t
        t: the value of t, starting at 0 for the first diffusion step
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        eta: the muliplication factor of the posterior distriution

        :returns
        a dict with following keys
        sample: a sample from the model
        pred_x0: a predition of x0
        """

        assert eta == 0.0, "Reverse ODE only for deterministic path"

        out = self.p_mean_variance(
            model=model, x=x, t=t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )

        eps = self._get_eps_from_x0(x, t, pred_x0=out["pred_x0"])
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = out["pred_x0"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return {
            "sample": mean_pred,
            "pred_x0": out["pred_x0"]
        }
    
    def ddim_sample_loop(self, model, shape, noise=None, clip_denoised=True, denoised_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        """
        Generates sample from the model using DDIM
        
        :params
        model: the model for predicting p(x_{t-1} | x_{t})
        shape: the shape of the image to be predicted
        noise: if specified the output of the revious timestep
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        device: if specified the device used for creting sample on. If not then uses the model device
        progress: if True, shows a tqdm progress bar

        :returns
        a dict with the samples from all the timesteps.
        """

        if device is None:
            device = next(model.parameters()).device
        
        assert isinstance(shape, (tuple, list))

        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)

        indices = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        samples = {}
        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model, img, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs, eta=eta
                )
                img = out["sample"]
                samples[i] = img

        return samples

    def _vb_terms_bpd(self, model, x0, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Get a term for the variational lower-bound
        
        :parmas
        model: the model used for diffusion
        x0: input image
        x_t: input image after t gaussian noises
        t: the current timestep
        clip_denoised: if True, clip the denoised signal [-1, 1]
        denoised_fn: if not None, a function which applies to the x_start prediction before it is used to sample. Applies before clip_denoised.
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.

        :returns
        a dict with the following keys
        - output: a shape [N] tensor of NLLs or KLs
        - pred_x0: the x_0 predictions
        """

        q_posterior = self.q_posterior_mean_variance(x_start=x0, x_t=x_t, t=t)
        out = self.p_mean_variance(
            model=model, x=x_t, t=t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )

        kl = kl_divergence(
            q_posterior["mean"], q_posterior["log_variance"], out["mean"], out["log_variance"]
        )
        # Converts from Nats to bits
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / torch.log(torch.tensor(2.0, dtype=kl.dtype))

        decoder_nll = -dicretixed_gaussian_log_likelihood(
            x0, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )

        decoder_nll = decoder_nll.mean(dim=list(range(1, len(decoder_nll.shape)))) / torch.log(torch.tensor(2.0, dtype=kl.dtype))

        output = torch.where((t == 0), decoder_nll, kl)

        return {
            "output": output,
            "pred_x0": out["pred_x0"]
        }
    
    def training_losses(self, model, x0, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, noise=None):
        """
        Calculate the training losses for a single timestep

        :params
        model: the model used for diffusion
        x0: the input images
        t: a batch of timesteps
        model_kwargs: if not None, a dict of extra keyword arguments to pass to the model. This can be used for conditioning.
        noise: if specified, the specific Gaussian noise to try to remove

        :returns
        a dict with loss containing a tensor of shape [N] giving the total loss
        the prediction of x0 and the x_t
        """

        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise=noise)

        terms = {}

        if self.model_loss_type == LossType.KL or self.model_loss_type == LossType.RESCALED_KL:
            output = self._vb_terms_bpd(
                model=model, x0=x0, x_t=x_t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
            )
            terms["loss"] = output["output"]
            pred_x0 = output["output"]

            if self.model_loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.model_loss_type == LossType.MSE or self.model_loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [ModelVarianceType.LEARNED, ModelVarianceType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])

                model_output, model_var_values = torch.split(model_output, C, dim=1)
                frozen_out = torch.cat([model_output.detach(), model_var_values], dim=1)
                def frozen_output(*args, r=frozen_out, **kwargs):
                    return r
                output = self._vb_terms_bpd(
                    model=frozen_output,
                    x0=x0, x_t=x_t, t=t, clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
                )

                terms["vb"] = output["output"]
                pred_x0 = output["pred_x0"]

                if self.model_loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0
            else:
                pred_x0 = {
                    ModelMeanType.PREVIOUS_X: self._get_x0_from_xprev(
                        x_t=x_t, t=t, x_prev=model_output
                    ),
                    ModelMeanType.START_X: model_output,
                    ModelMeanType.EPSILON: self._get_x0_from_epsilon(
                        x_t=x_t, t=t, eps=model_output
                    )
                }[self.model_mean_type]

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x0, x_t=x_t, t=t
                )["mean"],
                ModelMeanType.START_X: x0,
                ModelMeanType.EPSILON: noise
            }[self.model_mean_type]

            terms["mse"] = ((target - model_output) ** 2).mean(dim=list(range(1, len(target.shape))))
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            NotImplementedError(self.model_loss_type)
        
        return terms, pred_x0, x_t


