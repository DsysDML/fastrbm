from typing import Tuple

import h5py
import numpy as np
import torch
from rbms.io import load_params
from rbms.partition_function.ais import compute_partition_function_ais
from rbms.utils import get_saved_updates
from rbms.classes import EBM


def load_ais_traj_params(filename: str, index: int) -> Tuple[float, np.ndarray]:
    saved_updates = get_saved_updates(filename)
    with h5py.File(filename, "r") as f:
        log_z_init = f[f"update_{saved_updates[0]}"]["log_z"][()]  # float
        log_weights = f[f"update_{index}"]["log_weights"][()]
    return log_z_init, log_weights


def init_ais_traj_params(
    filename: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    map_model: dict[str, EBM],
):
    age = get_saved_updates(filename)[-1]
    params = load_params(
        filename, index=age, device=device, dtype=dtype, map_model=map_model
    )
    # For now we estimate the initial log Z using AIS temperature
    log_z = compute_partition_function_ais(
        num_chains=1000, num_beta=5000, params=params
    )
    log_weights = np.zeros(batch_size)
    save_ais_traj_params(
        filename=filename, age=age, log_z=log_z, log_weights=log_weights
    )


def save_ais_traj_params(
    filename: str, age: int, log_z: float, log_weights: np.ndarray
):
    with h5py.File(filename, "a") as f:
        f[f"update_{age}"]["log_z"] = log_z
        f[f"update_{age}"]["log_weights"] = log_weights
