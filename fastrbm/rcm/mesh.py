import numpy as np
import torch
from tqdm import tqdm

from fastrbm.rcm.lagrange import get_lagrange_multipliers

Tensor = torch.Tensor


def compute_mesh(
    U: torch.Tensor,
    n_pts_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    with_bias: bool = False,
    bias: float = 0.0,
    width: float = 1.0,
    dim_min=None,
    dim_max=None,
    bias_lim: torch.Tensor = None,
) -> torch.Tensor:
    num_visibles = U.shape[1]
    if dim_min is None:
        dim_min = np.ones(U.shape[0]) * -1
    if dim_max is None:
        dim_max = np.ones(U.shape[0])

    mesh = torch.meshgrid(
        *[
            torch.linspace(
                dim_min[i],
                dim_max[i],
                n_pts_dim,
                device=device,
                dtype=dtype,
            )
            for i in range(int(with_bias), U.shape[0])
        ],
        indexing="ij",
    )
    m = torch.vstack([elt.flatten() for elt in mesh]).T
    width = 1
    if with_bias:
        num_points = m.shape[0]
        if bias_lim is None:
            mesh_bias = bias + width / np.sqrt(num_visibles) * (
                torch.rand(num_points, device=device, dtype=dtype) * 2 - 1
            )
        else:
            mesh_bias = (
                torch.rand(num_points, device=device, dtype=dtype) * 2
                - 1 * (bias_lim[1] - bias_lim[0])
                + bias_lim[0]
            )
        m = torch.hstack([mesh_bias.unsqueeze(1), m])
    return m


def batched_lagrange_mult(
    m,
    U,
    err_threshold=1e-10,
    max_iter=10_000,
    batch_size=10_000,
    num_colors: int = 21,
    mesh_desc="coarse",
):

    num_points = m.shape[0]
    num_converged_points = 0
    # To avoid using too much memory
    if num_points > batch_size:
        n_batch = num_points // batch_size
        last_batch = False
        # Handle non divisible batch_size and num_points
        if num_points % batch_size != 0:
            last_batch = True
        all_m = []
        all_mu = []
        all_configurational_entropy = []
        with tqdm(range(n_batch + int(last_batch))) as pbar:
            pbar.set_description(f"Computing {mesh_desc} mesh")
            pbar.set_postfix_str(
                f"{num_converged_points}/{num_points} converged points"
            )
            for i in pbar:
                if last_batch and i == n_batch:
                    (
                        curr_m,
                        curr_mu,
                        curr_configurational_entropy,
                    ) = get_lagrange_multipliers(
                        m=m[batch_size * i :],
                        U=U,
                        err_threshold=err_threshold,
                        max_iter=max_iter,
                    )
                else:
                    (
                        curr_m,
                        curr_mu,
                        curr_configurational_entropy,
                    ) = get_lagrange_multipliers(
                        m=m[batch_size * i : batch_size * (i + 1)],
                        U=U,
                        err_threshold=err_threshold,
                        max_iter=max_iter,
                    )
                num_converged_points += len(curr_m)
                pbar.set_postfix_str(
                    f"{num_converged_points}/{num_points} converged points"
                )
                if len(curr_m) > 0:
                    all_m.append(curr_m)
                    all_mu.append(curr_mu)
                    all_configurational_entropy.append(curr_configurational_entropy)
            curr_m = torch.vstack(all_m)
            curr_mu = torch.vstack(all_mu)
            curr_configurational_entropy = torch.hstack(all_configurational_entropy)
    else:
        curr_m, curr_mu, curr_configurational_entropy = get_lagrange_multipliers(
            m, U, err_threshold=err_threshold, max_iter=max_iter
        )
    return curr_m, curr_mu, curr_configurational_entropy


def entropy_correction(
    m: Tensor, mu: Tensor, configurational_entropy: Tensor, U: Tensor
) -> Tensor:
    num_visibles = U.shape[1]
    device = m.device
    dtype = m.dtype
    counting = torch.zeros(len(m), device=device)
    with tqdm(enumerate(m), total=m.shape[0]) as pbar:
        pbar.set_description("Entropy correction")
        for id, m0 in pbar:
            num_samples = 1000
            iidx = (id * torch.ones(num_samples)).int()
            mu_full = (mu[iidx] @ U) * num_visibles**0.5
            x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
            p = 1 / (1 + torch.exp(-2 * mu_full))
            s_gen = 2 * (x < p) - 1
            m_gen = s_gen.float() @ U.T.float() / num_visibles**0.5
            indexes = torch.norm(m_gen - m0, dim=1) < 0.01
            counting[id] = torch.sum(indexes) / num_samples
    configurational_entropy = (
        configurational_entropy + torch.log(counting) / num_visibles
    )
    return configurational_entropy
