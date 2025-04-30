from typing import List, Optional, Tuple

import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.classes import EBM
from rbms.potts_bernoulli.classes import PBRBM
from torch import Tensor
from tqdm import tqdm

from fastrbm.trajectory.ptt_sampling_bernoulli import ptt_sampling
from fastrbm.rcm.rbm import sample_rbm
from fastrbm.utils import clone_dict, swap_chains


def sampling_step(
    list_params: List[EBM],
    chains: List[dict[str, Tensor]],
    rcm: Optional[dict],
    it_mcmc: int,
) -> List[dict[str, Tensor]]:
    """Performs it_mcmc sampling steps with all the models.

    Args:
        list_params List[RBM]: Saved models parameters.
        chains (List[Chain]): Previous configuration of the chains.
        it_mcmc (int): Number of steps to perform.

    Returns:
        List[Chain]: Updated chains.
    """
    # Sample from rbm
    use_rcm = rcm is not None
    n_chains = chains[0]["visible"].shape[0]
    if use_rcm:
        gen_rcm = sample_rbm(
            params=list_params[0],
            p_m=rcm["p_m"],
            mu=rcm["mu"],
            U=rcm["U"],
            num_samples=n_chains,
            device=chains[0]["visible"].device,
            dtype=torch.float32,
        )
        chains[0] = list_params[0].init_chains(n_chains, start_v=gen_rcm)
    for idx, params in enumerate(list_params[int(use_rcm) :], int(use_rcm)):
        chains[idx] = params.sample_state(chains=chains[idx], n_steps=it_mcmc)
    return chains


def swap_config_multi(
    params: List[EBM],
    chains: List[dict[str, Tensor]],
    index: Optional[List[Tensor]] = None,
    perform_swap: bool = True,
) -> Tuple[List[dict[str, Tensor]], Tensor, Optional[List[Tensor]]]:
    n_chains, L = chains[0]["visible"].shape
    n_rbms = len(params)
    acc_rate = torch.zeros(n_rbms - 1, device=chains[0]["visible"].device)
    for idx in range(n_rbms - 1):
        delta_energy = (
            -params[idx].compute_energy_visibles(chains[idx + 1]["visible"])
            + params[idx].compute_energy_visibles(chains[idx]["visible"])
            + params[idx + 1].compute_energy_visibles(chains[idx + 1]["visible"])
            - params[idx + 1].compute_energy_visibles(chains[idx]["visible"])
        )

        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )

        if index is not None:
            swapped_index_0 = torch.where(swap, index[idx + 1], index[idx])
            swapped_index_1 = torch.where(swap, index[idx], index[idx + 1])
            index[idx] = swapped_index_0
            index[idx + 1] = swapped_index_1

        acc_rate[idx] = swap.sum() / n_chains
        if perform_swap:
            chains[idx], chains[idx + 1] = swap_chains(
                chains[idx], chains[idx + 1], swap
            )

    return chains, acc_rate, index


def ptt_sampling(
    list_params: List[EBM],
    chains: List[dict[str, Tensor]],
    index: Optional[List[Tensor]],
    rcm: Optional[dict] = None,
    it_mcmc: int = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
) -> Tuple[List[dict[str, Tensor]], Tensor, Optional[List[Tensor]]]:
    assert len(list_params) == len(
        chains
    ), f"list_params and chains must have the same length, but got {len(list_params)} and {len(chains)}"
    if isinstance(list_params[0], BBRBM) and True:
        return ptt_sampling(
            list_params=list_params,
            chains=chains,
            index=index,
            it_mcmc=it_mcmc,
            rcm=rcm,
            increment=increment,
            show_pbar=show_pbar,
            show_acc_rate=show_acc_rate,
        )
    if show_pbar:
        pbar = tqdm(total=it_mcmc, leave=False)
    acc_rates = torch.zeros(len(list_params) - 1, device=chains[0]["visible"].device)
    for steps in range(0, it_mcmc, increment):
        if show_pbar:
            pbar.update(increment)
        chains, acc_rate, index = swap_config_multi(
            chains=chains, params=list_params, index=index
        )
        acc_rates += acc_rate
        chains = sampling_step(
            list_params=list_params,
            chains=chains,
            rcm=rcm,
            it_mcmc=increment,
        )
    acc_rates /= it_mcmc // increment
    if show_pbar:
        pbar.close()
    if show_acc_rate:
        print("acc_rate: ", acc_rates)
    return chains, acc_rates, index


def init_sampling(
    n_gen: int,
    list_params: List[EBM],
    start_v: Optional[Tensor] = None,
    it_mcmc: int = 1000,
    rcm: Optional[dict[str, Tensor]] = None,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    show_pbar: bool = True,
) -> List[dict[str, Tensor]]:
    use_dataset = start_v is not None
    use_rcm = (rcm is not None) and not use_dataset

    all_chains = []
    if show_pbar:
        pbar = tqdm(total=len(list_params))
        pbar.set_description("Initializing PTT chains")

    init_v = torch.bernoulli(
        torch.ones(n_gen, list_params[0].vbias.shape[0], device=device, dtype=dtype)
    )
    if use_rcm:
        init_v = sample_rbm(
            params=list_params[0],
            p_m=rcm["p_m"],
            mu=rcm["mu"],
            U=rcm["U"],
            num_samples=n_gen,
            device=device,
            dtype=dtype,
        )
    # Start every model from random permutations of the input dataset
    if use_dataset:
        perm_index = torch.randperm(start_v.shape[0])
        init_v = start_v[perm_index][:n_gen]

    for i, params in enumerate(list_params):
        chains = params.init_chains(num_samples=n_gen, start_v=init_v)

        # Iterate over the chains for some time
        chains = params.sample_state(n_steps=it_mcmc, chains=chains)

        init_v = chains["visible"]

        all_chains.append(clone_dict(chains))
        if show_pbar:
            pbar.update(1)
    return all_chains
