from typing import List, Optional, Tuple

import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.bernoulli_bernoulli.implement import _compute_energy_visibles
from torch import Tensor
from tqdm import tqdm

from fastrbm.rcm.rbm import sample_rbm


@torch.jit.script
def compute_energy_parallel(
    v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
):
    field = torch.bmm(v, vbias.view(vbias.shape[0], vbias.shape[1], 1)).squeeze()
    exponent = hbias.unsqueeze(1) + torch.bmm(v, weight_matrix)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(-1)


@torch.jit.script
def parallel_sampling(
    gibbs_steps: int, v: Tensor, weight_matrix: Tensor, vbias: Tensor, hbias: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    mh = torch.sigmoid(
        torch.bmm(v, weight_matrix) + hbias.view(hbias.shape[0], 1, hbias.shape[1])
    )
    h = torch.bernoulli(mh)
    mv = torch.zeros_like(v)
    for i in range(gibbs_steps):

        mv = torch.sigmoid(
            torch.bmm(h, weight_matrix.permute(0, 2, 1))
            + vbias.view(vbias.shape[0], 1, vbias.shape[1])
        )
        v = torch.bernoulli(mv)
        mh = torch.sigmoid(
            torch.bmm(v, weight_matrix) + hbias.view(hbias.shape[0], 1, hbias.shape[1])
        )
        h = torch.bernoulli(mh)
    return v, mv, h, mh


@torch.jit.script
def compute_delta_energy(energy_model_conf: Tensor) -> Tensor:
    return (
        -energy_model_conf[0][1]
        + energy_model_conf[0][0]
        + energy_model_conf[1][1]
        - energy_model_conf[1][0]
    )


@torch.jit.script
def swap_config_parallel(
    v: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    index: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    energies_next = compute_energy_parallel(
        v[1:], weight_matrix[:-1], vbias[:-1], hbias[:-1]
    )

    n_models = v.shape[0]
    acc_rate = torch.zeros(n_models - 1, device=v.device)
    n_chains = v.shape[1]
    energies = torch.zeros(2, 2, n_chains, device=v.device)
    energies[0][0] = _compute_energy_visibles(
        v[0], vbias[0], hbias[0], weight_matrix[0]
    )
    for i in range(n_models - 1):
        energies[0][1] = energies_next[i]

        energies[1][0] = _compute_energy_visibles(
            v[i], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
        )
        energies[1][1] = _compute_energy_visibles(
            v[i + 1], vbias[i + 1], hbias[i + 1], weight_matrix[i + 1]
        )
        delta_energy = compute_delta_energy(energies)
        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )
        energies[0][0] = torch.where(swap, energies[1][0], energies[1][1])
        acc_rate[i] = swap.sum() / n_chains

        if index is not None:
            index_save = index[i].clone()
            index[i] = torch.where(swap, index[i + 1], index_save)
            index[i + 1] = torch.where(swap, index_save, index[i + 1])
        swap = swap.view(-1, 1).repeat(1, v.shape[2])
        v_save = v[i].clone()
        v[i] = torch.where(swap, v[i + 1], v_save)
        v[i + 1] = torch.where(swap, v_save, v[i + 1])
    return v, acc_rate, index


def parallel_ptt_sampling(
    v: Tensor,
    weight_matrix: Tensor,
    vbias: Tensor,
    hbias: Tensor,
    it_mcmc: int,
    increment: int = 10,
    rcm: Optional[dict[str, Tensor]] = None,
    show_pbar: bool = True,
    index: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    acc_rates = torch.zeros(v.shape[0] - 1, device=v.device)
    use_rcm = rcm is not None
    if show_pbar:
        pbar = tqdm(total=it_mcmc, leave=False)
    for steps in range(0, it_mcmc, increment):
        if show_pbar:
            pbar.update(increment)
        v, acc_rate, index = swap_config_parallel(v, weight_matrix, vbias, hbias, index)
        acc_rates += acc_rate
        if use_rcm:
            v[0] = sample_rbm(
                p_m=rcm["p_m"],
                mu=rcm["mu"],
                U=rcm["U"],
                num_samples=v.shape[1],
                device=v.device,
                dtype=v.dtype,
            )
        v, mv, h, mh = parallel_sampling(increment, v, weight_matrix, vbias, hbias)
    acc_rates /= it_mcmc // increment
    if show_pbar:
        pbar.close()
    return v, h, mv, mh, acc_rates, index


def ptt_sampling(
    list_params: List[BBRBM],
    chains: List[dict[str, Tensor]],
    index: Optional[List[Tensor]],
    it_mcmc: int = None,
    rcm: Optional[dict[str, Tensor]] = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
):
    weight_matrix = torch.stack([p.weight_matrix for p in list_params])
    vbias = torch.stack([p.vbias for p in list_params])
    hbias = torch.stack([p.hbias for p in list_params])
    if index is not None:
        index = torch.stack([idx for idx in index])
    v = torch.stack([c["visible"] for c in chains])
    v, h, mv, mh, acc_rates, index = parallel_ptt_sampling(
        v, weight_matrix, vbias, hbias, it_mcmc, increment, rcm, show_pbar, index
    )
    ret_index = None
    if index is not None:
        ret_index = []
        for i in range(len(index)):
            ret_index.append(index[i])
    for i, c in enumerate(chains):
        c["visible"] = v[i].clone()
        c["hidden"] = h[i].clone()
        c["visible_mag"] = mv[i].clone()
        c["hidden_mag"] = mh[i].clone()
    if show_acc_rate:
        print("acc_rate: ", acc_rates)
    return chains, acc_rates, ret_index
