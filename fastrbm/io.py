from typing import List

import h5py
import torch
from rbms.classes import EBM
from rbms.io import load_params
from rbms.utils import get_flagged_updates
from torch import Tensor


def load_rcm(
    filename: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor] | None:
    """Load the RCM saved in the given RBM h5 archive.

    Parameters
    ----------
    filename : str
        Path to the h5 archive
    device : torch.device
        PyTorch device on which to load the RCM
    dtype: torch.dtype
        Dtype for the RCM

    Returns
    ----------
    dict[str, Tensor] | None
        The RCM if it exists, otherwise None
    """
    rcm = None
    with h5py.File(filename, "r") as f:
        if "rcm" in f.keys():
            U = torch.from_numpy(f["rcm"]["U"][()]).to(device=device, dtype=dtype)
            m = torch.from_numpy(f["rcm"]["m"][()]).to(device=device, dtype=dtype)
            mu = torch.from_numpy(f["rcm"]["mu"][()]).to(device=device, dtype=dtype)
            p_m = torch.from_numpy(f["rcm"]["pdm"][()]).to(device=device, dtype=dtype)
            rcm = {"U": U, "m": m, "mu": mu, "p_m": p_m}
    return rcm


def load_params_ptt(
    filename: str, device: torch.device, dtype: torch.dtype
) -> List[EBM]:
    ptt_updates = get_flagged_updates(filename, "ptt")
    list_params = []
    for upd in ptt_updates:
        params = load_params(filename, index=upd, device=device, dtype=dtype)
        list_params.append(params)
    return list_params
