import torch

def kl_divergence(mean1, logvar1, mean2, logvar2):
    """
    Computes the KL divergence between the two gaussian distributions
    """

    logvar1, logvar2 = [x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in [logvar1, logvar2]]

    loss = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar2 - logvar1) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return loss

def approximate_standard_normal_cdf(x):
    """
    A fast apprximation of the cummalitive distribution function of the standard normal
    """

    return 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(1.0, dtype=x.dtype) / torch.pi)) * (x + 0.044715 * torch.pow(x, 3)))

def dicretixed_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a gaussian distribution discretizing to the given image

    :params
    x: target image. It is assumed that this was uint8 values, rescalled to the range [-1, 1]
    means: the gaussian mean tensor
    log_scales: the gaussian log stddev tensor

    :returns
    a tensor like x of log probabilities
    """

    centered_x = x - means
    inv_stddev = torch.exp(-log_scales)

    plus_in =  inv_stddev * (centered_x + 1.0 / 255.0)
    cdf_plus = approximate_standard_normal_cdf(plus_in)

    minus_in = inv_stddev * (centered_x - 1.0 / 255.0)
    cdf_min = approximate_standard_normal_cdf(minus_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_minus = torch.log((1 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_minus, torch.log(cdf_delta.clamp(min=1e-12)))
    )

    return log_probs