import numpy as np
import torch
from rbms.potts_bernoulli.tools import get_covariance_matrix
from torch import Tensor


def compute_U(
    M: Tensor,
    weights: Tensor,
    intrinsic_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    with_bias: bool = False,
):
    _, num_visibles = M.shape
    weights /= weights.sum()
    cov_data = get_covariance_matrix(M, weights, device=device, center=with_bias).to(
        device=device, dtype=dtype
    )

    _, V_dataT = torch.lobpcg(cov_data, k=intrinsic_dimension)
    u = V_dataT
    u = u[:, :intrinsic_dimension]

    mean_value = (M.T @ weights).squeeze()
    z = 0
    if with_bias:
        bias_vector = mean_value - u.T @ mean_value @ u.T
        z = bias_vector.norm()
        bias_vector /= z
        u = torch.hstack([bias_vector.unsqueeze(1), u])
    return u, z / np.sqrt(num_visibles)
