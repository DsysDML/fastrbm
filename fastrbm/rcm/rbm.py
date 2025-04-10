from typing import Tuple

import numpy as np
import torch
from rbms.custom_fn import log2cosh
from torch import Tensor
from tqdm import tqdm

from fastrbm.rcm.log import build_log_string_train


def get_energy_rbm(
    m: Tensor,
    Edm: Tensor,
    vbias: Tensor,
    configurational_entropy: Tensor,
) -> Tensor:
    return Edm - m @ vbias - configurational_entropy


def get_ll_rbm(
    configurational_entropy: Tensor,
    data: Tensor,
    m: Tensor,
    W: Tensor,
    hbias: Tensor,
    vbias: Tensor,
    U: Tensor,
    num_visibles: int,
    return_logZ: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    num_samples = data.shape[0]
    proj_W = U @ W
    proj_vbias = (U @ vbias) / num_visibles**0.5
    Edm = -log2cosh((num_visibles**0.5) * (m @ proj_W) - hbias).sum(1) / num_visibles
    Fs = (data @ proj_W) * (num_visibles**0.5) - hbias.unsqueeze(0)
    Eds = -log2cosh(Fs).sum() / num_visibles
    Eds /= num_samples

    sample_energy = Eds - data @ proj_vbias
    # compute Z
    LL = -num_visibles * torch.mean(sample_energy)
    m_energy = get_energy_rbm(
        m=m,
        Edm=Edm,
        vbias=proj_vbias,
        configurational_entropy=configurational_entropy,
    )
    F = m_energy
    F0 = -configurational_entropy
    logZ = torch.logsumexp(-num_visibles * F, 0)
    logZ0 = torch.logsumexp(-num_visibles * F0, 0)
    logZ00 = num_visibles * np.log(2)
    logZ -= logZ0 - logZ00
    if return_logZ:
        return LL - logZ, logZ
    return LL - logZ


def get_proba_rbm(
    m: Tensor,
    configurational_entropy: Tensor,
    U: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    W: Tensor,
    return_logZ: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    num_visibles = vbias.shape[0]
    num_points = m.shape[0]
    pdm = torch.zeros(num_points)
    proj_W = U @ W
    proj_vbias = (U @ vbias) / num_visibles**0.5
    energy_m = (
        -log2cosh(num_visibles**0.5 * (m @ proj_W) - hbias).sum(1) / num_visibles
        - m @ proj_vbias
        - configurational_entropy
    )
    Fmin = energy_m.min()

    Z = torch.exp(-num_visibles * (energy_m - Fmin)).sum()
    pdm = torch.exp(-num_visibles * (energy_m - Fmin)) / Z
    if return_logZ:
        return pdm, torch.log(Z)
    return pdm


def sample_rbm(
    p_m: Tensor,
    mu: Tensor,
    U: Tensor,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    num_points, intrinsic_dimension = mu.shape
    num_visibles = U.shape[1]
    cdf = torch.zeros(num_points + 1, device=device, dtype=dtype)
    cdf[1:] = torch.cumsum(p_m, 0)
    x = torch.rand(num_samples, device=device, dtype=dtype)
    idx = torch.searchsorted(sorted_sequence=cdf, input=x) - 1
    mu_full = (mu[idx] @ U) * num_visibles**0.5  # n_samples x Nv
    x = torch.rand((num_samples, num_visibles), device=device, dtype=dtype)
    p = 1 / (1 + torch.exp(-2 * mu_full))  # n_samples x Nv
    s_gen = 2 * (x < p) - 1
    return s_gen


def compute_neg_grad_bias(m: Tensor, pdm: Tensor) -> Tensor:
    return m.T @ pdm


def compute_pos_grad_bias(proj_train: Tensor, weights: Tensor) -> Tensor:
    return (proj_train * weights).sum(0) / weights.sum()


def finetune_bias(
    proj_train: Tensor,
    proj_test: Tensor,
    weights_train: Tensor,
    m: Tensor,
    mu: Tensor,
    configurational_entropy: Tensor,
    U: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    W: Tensor,
    learning_rate: float,
    max_iter: int,
    adapt: bool,
    min_learning_rate: float,
    stop_ll: float,
    with_bias: bool = False,
):
    pbar = tqdm(range(max_iter))
    num_visibles = U.shape[1]
    grad_pos = compute_pos_grad_bias(proj_train=proj_train, weights=weights_train)
    count_lr = 0
    curr_ll = get_ll_rbm(
        configurational_entropy=configurational_entropy,
        data=proj_train,
        m=m,
        W=W,
        hbias=hbias,
        vbias=vbias,
        U=U,
        num_visibles=num_visibles,
        return_logZ=False,
        with_bias=with_bias,
    )
    best_ll = curr_ll
    all_train_ll = []
    all_test_ll = []
    prev_test_ll = 0
    best_vbias = vbias
    for n_iter in pbar:
        pdm = get_proba_rbm(
            m=m,
            configurational_entropy=configurational_entropy,
            U=U,
            vbias=vbias,
            hbias=hbias,
            W=W,
            return_logZ=False,
        )
        grad_neg = compute_neg_grad_bias(m=m, pdm=pdm)
        grad = grad_pos - grad_neg
        proj_vbias = (U @ vbias) / num_visibles**0.5
        proj_vbias += learning_rate * grad
        if with_bias:
            proj_vbias[0] = pdm @ mu[:, 0]
        vbias = proj_vbias @ U * num_visibles**0.5
        if n_iter % 100 == 0:
            new_ll = get_ll_rbm(
                configurational_entropy=configurational_entropy,
                data=proj_train,
                m=m,
                W=W,
                hbias=hbias,
                vbias=vbias,
                U=U,
                num_visibles=num_visibles,
                return_logZ=False,
                with_bias=with_bias,
            )
            if new_ll > curr_ll:
                learning_rate *= 1.0 + 0.02 * adapt
                count_lr += 1
            else:
                learning_rate *= 1.0 - 0.1 * adapt
            learning_rate = max(min_learning_rate, learning_rate)
            if new_ll > best_ll:
                best_vbias = torch.clone(vbias)
                best_ll = new_ll
            curr_ll = new_ll
            if n_iter % 1000 == 0:
                new_test_ll = get_ll_rbm(
                    configurational_entropy=configurational_entropy,
                    data=proj_test,
                    m=m,
                    W=W,
                    hbias=hbias,
                    vbias=vbias,
                    U=U,
                    num_visibles=num_visibles,
                    return_logZ=False,
                    with_bias=with_bias,
                )
                all_train_ll.append(new_ll.item())
                all_test_ll.append(new_test_ll.item())
                grad_vbias_norm = torch.norm(grad_pos - grad_neg)
                log_string = build_log_string_train(
                    train_ll=new_ll.item(),
                    test_ll=new_test_ll.item(),
                    n_iter=n_iter,
                    mean_q=0.0,
                    grad_vbias_norm=grad_vbias_norm.item(),
                    curr_lr=learning_rate,
                    count=count_lr,
                )
                count_lr = 0
                pbar.write(log_string)
                if torch.abs(prev_test_ll - new_test_ll) < 0.01 * stop_ll:
                    break
                prev_test_ll = prev_test_ll * 0.05 + new_test_ll * 0.95
    return best_vbias, all_train_ll, all_test_ll
