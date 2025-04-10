from typing import List, Optional

import numpy as np
import torch
from rbms.classes import EBM
from rbms.partition_function.ais import compute_partition_function_ais
from torch import Tensor


def compute_partition_function_ptt(
    list_params: List[EBM],
    list_chains: List[dict[str, Tensor]],
    log_z_init: Optional[float] = None,
) -> Tensor:
    if log_z_init is None:
        # Estimate the first log Z using AIS
        # Should work if this distribution is not multimodal
        log_z_init = compute_partition_function_ais(
            num_chains=5000,
            num_beta=1000,
            params=list_params[0],
        )

    logz = log_z_init
    logZ = torch.zeros(len(list_params))
    logZ[0] = log_z_init
    for idx in range(len(list_params) - 1):
        E0 = list_params[idx].compute_energy_visibles(list_chains[idx]["visible"])
        E1 = list_params[idx + 1].compute_energy_visibles(list_chains[idx]["visible"])
        c0 = torch.logsumexp(-E1 + E0, dim=0) - np.log(
            list_chains[idx]["visible"].shape[0]
        )
        logz += c0
        logZ[idx + 1] = logz
    return logZ


def update_partition_function_ptt(
    params_curr: EBM,
    params_prev: EBM,
    chains_curr: dict[str, Tensor],
    log_z_prev: float,
):
    energy_prev = params_prev.compute_energy_visibles(chains_curr["visible"])
    energy_curr = params_curr.compute_energy_visibles(chains_curr["visible"])
    update_log_z = torch.logsumexp(-energy_curr + energy_prev, dim=0) - np.log(
        chains_curr["visible"].shape[0]
    )
    return log_z_prev + update_log_z
