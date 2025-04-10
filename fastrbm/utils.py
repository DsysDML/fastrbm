from typing import List, Tuple

import torch
from rbms.classes import EBM
from rbms.io import load_params
from rbms.utils import get_flagged_updates
from torch import Tensor
from torch.nn.functional import softmax


def clone_dict(d: dict[str, Tensor]) -> dict[str, Tensor]:
    res = {}
    for k in d.keys():
        res[k] = d[k].clone()
    return res


def add_machine(
    list_params: List[EBM], chains: List[dict[str, Tensor]], it_mcmc: int
) -> Tuple[List[dict[str, Tensor]], List[EBM]]:
    list_params.insert(-1, list_params[-1].clone())
    chains.insert(-1, clone_dict(chains[-1]))
    chains[-1] = list_params[-1].sample_state(n_steps=it_mcmc, chains=chains[-1])
    return chains, list_params


def load_params_ptt(
    filename: str, map_model: dict[str, EBM], device: torch.device, dtype: torch.dtype
) -> List[EBM]:
    ptt_updates = get_flagged_updates(filename, "ptt")
    list_params = []
    for upd in ptt_updates:
        params = load_params(
            filename, index=upd, device=device, dtype=dtype, map_model=map_model
        )
        list_params.append(params)
    return list_params


def compute_ess(logit_weights: Tensor) -> Tensor:
    """Computes the Effective Sample Size of the chains.

    Args:
        logit_weights: minus log-weights of the chains.
    """
    lwc = logit_weights - logit_weights.min()
    numerator = torch.square(torch.mean(torch.exp(-lwc)))
    denominator = torch.mean(torch.exp(-2.0 * lwc))

    return numerator / denominator


@torch.jit.script
def swap(x: torch.Tensor, y: torch.Tensor, idx_swap: Tensor):
    t = x[idx_swap]
    x[idx_swap] = y[idx_swap]
    y[idx_swap] = t


@torch.jit.script
def swap_chains_inplace(
    chain_1: dict[str, Tensor], chain_2: dict[str, Tensor], idx: Tensor
) -> None:
    """
    Swap elements between two dict[str, Tensor] at specified indices.

    Args:
        chain_1 (dict[str, Tensor]): First chain.
        chain_2 (dict[str, Tensor]): Second chain.
        idx (Tensor): Tensor of indices specifying which elements to swap between the chains.

    Returns:
        Tuple[dict[str, Tensor], dict[str, Tensor]]: Modified chains after swapping.
    """
    swap(chain_1["weights"], chain_2["weights"], idx)
    swap(chain_1["visible"], chain_2["visible"], idx)
    swap(chain_1["hidden"], chain_2["hidden"], idx)
    swap(chain_1["visible_mag"], chain_2["visible_mag"], idx)
    swap(chain_1["hidden_mag"], chain_2["hidden_mag"], idx)


@torch.jit.script
def swap_chains(
    chain_1: dict[str, Tensor], chain_2: dict[str, Tensor], idx: Tensor
) -> Tuple[dict[str, Tensor], dict[str, Tensor]]:
    new_chain_1 = dict()
    new_chain_2 = dict()

    new_chain_1["weights"] = torch.where(
        idx, chain_2["weights"].squeeze(), chain_1["weights"].squeeze()
    ).unsqueeze(-1)
    new_chain_2["weights"] = torch.where(
        idx, chain_1["weights"].squeeze(), chain_2["weights"].squeeze()
    ).unsqueeze(-1)

    idx_vis = idx.unsqueeze(1).repeat(1, chain_1["visible"].shape[1])

    # if len(chain_1["visible_mag"].shape) > len(chain_1["visible"].shape):
    #     idx_vis_mean = idx_vis.repeat(1, chain_1["visible_mag"].shape[2]).reshape(
    #         chain_1["visible_mag"].shape
    #     )
    # else:
    #     idx_vis_mean = idx_vis

    # idx_hid = idx.unsqueeze(1).repeat(1, chain_1["hidden"].shape[1])

    new_chain_1["visible"] = torch.where(
        idx_vis, chain_2["visible"], chain_1["visible"]
    )
    new_chain_2["visible"] = torch.where(
        idx_vis, chain_1["visible"], chain_2["visible"]
    )

    # new_chain_1["visible_mag"] = torch.where(
    #     idx_vis_mean, chain_2["visible_mag"], chain_1["visible_mag"]
    # )
    # new_chain_2["visible_mag"] = torch.where(
    #     idx_vis_mean, chain_1["visible_mag"], chain_2["visible_mag"]
    # )

    # new_chain_1["hidden"] = torch.where(idx_hid, chain_2["hidden"], chain_1["hidden"])
    # new_chain_2["hidden"] = torch.where(idx_hid, chain_1["hidden"], chain_2["hidden"])

    # new_chain_1["hidden_mag"] = torch.where(
    #     idx_hid, chain_2["hidden_mag"], chain_1["hidden_mag"]
    # )
    # new_chain_2["hidden_mag"] = torch.where(
    #     idx_hid, chain_1["hidden_mag"], chain_2["hidden_mag"]
    # )

    return new_chain_1, new_chain_2


@torch.jit.script
def systematic_resampling(
    chains: dict[str, Tensor], log_weights: Tensor
) -> dict[str, Tensor]:
    """Performs the systematic resampling of the chains according to their relative weight and
    sets the logit_weights back to zero.

    Args:
        chains (Chain): Chains.

    Returns:
        Chain: Resampled chains.
    """
    num_chains = chains["visible"].shape[0]
    device = chains["visible"].device
    weights = softmax(-log_weights, -1)
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    chains["visible"] = torch.repeat_interleave(chains["visible"], counts, dim=0)
    chains["hidden"] = torch.repeat_interleave(chains["hidden"], counts, dim=0)

    return chains
