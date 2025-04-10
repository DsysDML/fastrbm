import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import RBM
from torch import Tensor


def sample_rcm_bernoulli(
    p_m: Tensor,
    mu: Tensor,
    U: Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    num_visibles = U.shape[1]
    cdf = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv
    x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
    p = 1 / (1 + torch.exp(-2 * mu_full))  # n_samples x Nv
    # We want {0,1} samples
    s_gen = x < p
    return s_gen


def sample_rcm(
    params: RBM,
    p_m: Tensor,
    mu: Tensor,
    U: Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
):
    if isinstance(params, BBRBM):
        return sample_rcm_bernoulli(
            p_m=p_m, mu=mu, U=U, num_samples=num_samples, device=device, dtype=dtype
        )
    raise ValueError(f"This RBM type is not supported for RCM: {type(params)}")
