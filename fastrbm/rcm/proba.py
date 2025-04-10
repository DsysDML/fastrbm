from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from fastrbm.rcm.features import evaluate_features


def compute_p_rcm(
    m: Tensor,
    configurational_entropy: Tensor,
    features_rcm: Tensor,
    vbias: Tensor,
    q: Tensor,
    num_visibles: int,
) -> Tensor:
    """Computes an estimation of the density learned by the RCM on the discretization points.

    Parameters
    ----------
    m : Tensor
        Discretization points. (n_points, n_dim)
    configurational_entropy : Tensor
        Configurational entropy for the discretization points. (n_points,)
    features_rcm : Tensor
        Evaluation of the features of the RCM on the discretization points. (n_points,)
    vbias : Tensor
        Visible bias of the RCM. (n_dim,)
    q : Tensor
        Hyperplanes weights. (n_feat,)
    num_visibles : int
        Dimension of the original dataset.

    Returns
    -------
    Tensor
        Estimation of the density. (n_points,)
    """
    F = -configurational_entropy - m @ vbias - features_rcm.T @ q
    p_m = torch.exp(-num_visibles * (F - F.min()))
    Z = p_m.sum()
    return p_m / Z


def get_energy_coulomb(
    sample: Tensor,
    features: Tensor,
    q: Tensor,
    vbias: Tensor,
) -> Tensor:
    """Compute the energy of the RCM for each of the sample points.

    Parameters
    ----------
    sample : Tensor
        Sample points. (n_points, n_dim)
    features : Tensor
        Features of the RCM. (n_feat, n_dim+1)
    q : Tensor
        Hyperplanes weights. (n_feat,)
    vbias : Tensor
        Visible bias of the RCM.

    Returns
    -------
    Tensor
        Energy of the RCM. (n_points,)
    """
    eval_features = evaluate_features(features=features, sample=sample)
    return -q @ eval_features - vbias @ sample.T


def get_ll_coulomb(
    configurational_entropy: Tensor,
    data: Tensor,
    m: Tensor,
    features: Tensor,
    q: Tensor,
    vbias: Tensor,
    U: Tensor,
    num_visibles: int,
    return_logZ: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """Compute the log-likelihood of the RCM on the data.

    Parameters
    ----------
    configurational_entropy : Tensor
        Configurational_entropy of the discretization points. (n_points)
    data : Tensor
        Data points. (n_data, n_dim)
    m : Tensor
        Discretization points. (n_points, n_dim)
    features : Tensor
        Features of the RCM. (n_feat, n_dim+1)
    q : Tensor
        Hyperplanes weights. (n_feat,)
    vbias : Tensor
        Visible bias of the RCM. (n_dim)
    num_visibles : int
        Dimension of the original dataset,

    Returns
    -------
    Tensor
        Log-likelihood
    """
    sample_energy = get_energy_coulomb(
        sample=data,
        features=features,
        q=q,
        vbias=vbias,
    )

    # compute Z
    LL = -num_visibles * torch.mean(sample_energy)
    m_energy = get_energy_coulomb(sample=m, features=features, q=q, vbias=vbias)
    F = m_energy - configurational_entropy
    F0 = -configurational_entropy

    logZ = torch.logsumexp(-num_visibles * F, 0)
    logZ0 = torch.logsumexp(-num_visibles * F0, 0)
    logZ00 = num_visibles * np.log(2)
    logZ -= logZ0 - logZ00
    if return_logZ:
        return LL - logZ, logZ
    return LL - logZ
