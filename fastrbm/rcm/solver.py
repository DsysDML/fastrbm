from typing import Tuple

import torch
from torch import Tensor
from tqdm import tqdm

from fastrbm.rcm.features import evaluate_features
from fastrbm.rcm.log import build_log_string_train
from fastrbm.rcm.proba import compute_p_rcm, get_ll_coulomb


@torch.jit.script
def compute_pos_grad(
    proj_train: Tensor,
    features: Tensor,
    weights: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute the positive term of the gradient for the visible bias and the hyperplanes weights.

    Parameters
    ----------
    proj_train : Tensor
        The projected training dataset. (n_samples, n_dim)
    features : Tensor
        The features of the RCM. (n_feat, n_dim+1)

    Returns
    -------
    Tensor
        Positive term of the gradient for the visible bias. (n_dim,)

    Tensor
        Positive term of the gradient for the hyperplanes weights. (n_feat,)
    """
    grad_vbias_pos = (proj_train * weights.unsqueeze(1)).sum(0) / weights.sum()
    grad_q_pos = (
        evaluate_features(features=features, sample=proj_train)
        * weights.squeeze().unsqueeze(0)
    ).sum(1) / weights.sum()
    return grad_vbias_pos, grad_q_pos


@torch.jit.script
def compute_neg_grad(
    m: Tensor,
    mu: Tensor,
    features_rcm: Tensor,
    p_m: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Compute the negative term of the gradient for the visible bias and the hyperplanes weights.

    Parameters
    ----------
    m : Tensor
        The discretization points. (n_points, n_dim)
    features_rcm : Tensor
        The features of the model evaluated on the discretization points. (n_points,)
    p_m : Tensor
        The probability density estimated by the RCM in each of the discretization points. (n_points,)

    Returns
    -------
    Tensor
        Negative term of the gradient for the visible bias. (n_dim,)

    Tensor
        Negative term of the gradient for the hyperplanes weights. (n_feat,)
    """
    grad_vbias_neg = m.T @ p_m
    grad_q_neg = features_rcm @ p_m
    return grad_vbias_neg, grad_q_neg


def compute_hessian(
    prev_hessian: Tensor,
    m: Tensor,
    p_m: Tensor,
    grad_vbias_neg: Tensor,
    grad_q_neg: Tensor,
    features_rcm: Tensor,
    smooth_rate: float,
) -> Tensor:
    """Compute an estimate of the Hessian for the whole set of parameters. The Hessian is then interpolated with previous one based on smooth rate.

    Parameters
    ----------
    prev_hessian : Tensor
        Previous Hessian. (n_dim+n_feat, n_dim+n_feat)
    m : Tensor
        The discretization points. (n_points, n_dim)
    p_m : Tensor
        The probability density estimated by the RCM in each of the discretization points. (n_points,)
    grad_vbias_neg : Tensor
        The negative term of the visible bias gradient. (n_dim,)
    grad_q_neg : Tensor
        The negative term of the hyperplanes weights gradient. (n_feat,)
    features_rcm : Tensor
        The features of the model evaluated on the discretization points. (n_points)
    smooth_rate : float
        The interpolation strength. `1` means no interpolation and `0` keep the previous Hessian.

    Returns
    -------
    Tensor
        The new Hessian (n_dim+n_feat, n_dim+n_feat)
    """
    device = p_m.device
    R = torch.hstack([m, features_rcm.T])
    xr = torch.cat([grad_vbias_neg, grad_q_neg])
    # Need to use sparse matrices to avoid memory overflow while keeping a vectorized computation.
    tmp = torch.sparse.mm(
        torch.sparse.spdiags(
            p_m.cpu(), offsets=torch.tensor([0]), shape=(p_m.shape[0], p_m.shape[0])
        ).to(device),
        R,
    ).T
    new_hessian = tmp @ R - xr.unsqueeze(1) @ xr.unsqueeze(0)
    return (1 - smooth_rate) * prev_hessian + smooth_rate * new_hessian


@torch.jit.script
def update_parameters(
    vbias: Tensor,
    q: Tensor,
    grad_vbias_neg: Tensor,
    grad_vbias_pos: Tensor,
    grad_q_neg: Tensor,
    grad_q_pos: Tensor,
    inverse_hessian: Tensor,
    learning_rate: float,
) -> Tuple[Tensor, Tensor]:
    """Updates the parameters of the RCM using a quasi-Newton method.

    Parameters
    ----------
    vbias : Tensor
        Visible bias. (n_dim,)
    q : Tensor
        Hyperplanes weights (n_feat,)
    grad_vbias_neg : Tensor
        Negative term of the visible bias gradient. (n_dim,)
    grad_vbias_pos : Tensor
        Positive term of the visible bias gradient. (n_dim,)
    grad_q_neg : Tensor
        Negative term of the hyperplanes weights gradient. (n_dim,)
    grad_q_pos : Tensor
        Positive term of the hyperplanes weigths gradient. (n_dim,)
    inverse_hessian : Tensor
        Inverse of the estimate of the Hessian for the trainable parameters. (n_dim+n_feat, n_dim+n_feat)
    learning_rate : float
        Learning rate.

    Returns
    -------
    Tensor
        Updated visible bias. (n_dim,)
    Tensor
        Updated hyperplanes weigths. (n_feat,)
    """
    intrinsic_dimension = vbias.shape[0]
    params = torch.cat([vbias, q])

    grad_params = torch.cat([grad_vbias_pos - grad_vbias_neg, grad_q_pos - grad_q_neg])
    params += inverse_hessian @ grad_params * learning_rate
    vbias = params[:intrinsic_dimension]
    q = params[intrinsic_dimension:]
    q[q < 0] = 0
    return vbias, q


def train_rcm(
    proj_train: Tensor,
    proj_test: Tensor,
    weights_train: Tensor,
    weights_test: Tensor,
    m: Tensor,
    mu: Tensor,
    configurational_entropy: Tensor,
    features: Tensor,
    U: Tensor,
    max_iter: int,
    num_visibles: int,
    learning_rate: float,
    adapt: bool,
    min_learning_rate: float,
    smooth_rate: float,
    stop_ll: float,
    total_iter: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
    num_features = features.shape[0]
    intrinsic_dimension = m.shape[1]
    q = torch.zeros(num_features, device=device, dtype=dtype)
    vbias = torch.zeros(intrinsic_dimension, device=device, dtype=dtype)
    vbias = proj_train.mean(0).to(device=device, dtype=dtype)
    features_rcm = evaluate_features(features=features, sample=m)

    prev_test_ll = 0
    curr_ll = get_ll_coulomb(
        configurational_entropy=configurational_entropy,
        data=proj_train,
        m=m,
        features=features,
        q=q,
        vbias=vbias,
        U=U,
        num_visibles=num_visibles,
        return_logZ=False,
    )
    best_ll = curr_ll
    best_q = q.clone()
    best_vbias = vbias.clone()
    grad_vbias_pos, grad_q_pos = compute_pos_grad(
        proj_train=proj_train,
        features=features,
        weights=weights_train,
    )
    print(f"LL start: {curr_ll}")

    count_lr = 0

    hessian = torch.eye(num_features + intrinsic_dimension, device=device, dtype=dtype)
    inverse_hessian = torch.eye(
        num_features + intrinsic_dimension, device=device, dtype=dtype
    )
    pbar = tqdm(range(max_iter))
    header = " num iter | train ll | test ll  |  mean q  | |\u2207vbias| | curr lr  | count +lr"
    pbar.write(header)
    all_train_ll = []
    all_test_ll = []
    cpt = 0
    for n_iter in pbar:
        p_m = compute_p_rcm(
            m=m,
            configurational_entropy=configurational_entropy,
            features_rcm=features_rcm,
            vbias=vbias,
            q=q,
            num_visibles=num_visibles,
        )
        grad_vbias_neg, grad_q_neg = compute_neg_grad(
            m=m, mu=mu, features_rcm=features_rcm, p_m=p_m
        )
        tmp_prev_ll = get_ll_coulomb(
            configurational_entropy=configurational_entropy,
            data=proj_train,
            m=m,
            features=features,
            q=q,
            vbias=vbias,
            U=U,
            num_visibles=num_visibles,
        )
        vbias, q = update_parameters(
            vbias=vbias,
            q=q,
            grad_vbias_neg=grad_vbias_neg,
            grad_vbias_pos=grad_vbias_pos,
            grad_q_neg=grad_q_neg,
            grad_q_pos=grad_q_pos,
            inverse_hessian=inverse_hessian,
            learning_rate=learning_rate,
        )
        tmp_new_ll = get_ll_coulomb(
            configurational_entropy=configurational_entropy,
            data=proj_train,
            m=m,
            features=features,
            q=q,
            vbias=vbias,
            U=U,
            num_visibles=num_visibles,
        )
        cpt += int(tmp_new_ll - tmp_prev_ll >= 0)
        if n_iter % 100 == 0:
            new_ll = get_ll_coulomb(
                configurational_entropy=configurational_entropy,
                data=proj_train,
                m=m,
                features=features,
                q=q,
                vbias=vbias,
                U=U,
                num_visibles=num_visibles,
            )
            if new_ll > curr_ll:
                learning_rate *= 1.0 + 0.02 * adapt
                count_lr += 1
            else:
                learning_rate *= 1.0 - 0.1 * adapt
            learning_rate = max(min_learning_rate, learning_rate)
            if new_ll > best_ll:
                best_q = torch.clone(q)
                best_vbias = torch.clone(vbias)
                best_ll = new_ll
            curr_ll = new_ll
            hessian = compute_hessian(
                prev_hessian=hessian,
                m=m,
                p_m=p_m,
                grad_vbias_neg=grad_vbias_neg,
                grad_q_neg=grad_q_neg,
                features_rcm=features_rcm,
                smooth_rate=smooth_rate,
            )
            # Lstsq is not precise enough -> the training diverges at some point
            # inverse_hessian = torch.linalg.lstsq(
            #     hessian, torch.eye(hessian.shape[0], device=device, dtype=dtype)
            # ).solution

            # pinv is slow
            # inverse_hessian = torch.linalg.pinv(hessian, rtol=1e-4)

            # Regularized inversion seems to remain the best
            inverse_hessian = torch.linalg.inv(
                hessian
                + torch.diag(
                    torch.ones(hessian.shape[0], device=device, dtype=dtype) * 1e-5
                )
            )
            if n_iter % 1000 == 0:
                new_test_ll = get_ll_coulomb(
                    configurational_entropy=configurational_entropy,
                    data=proj_test,
                    m=m,
                    features=features,
                    q=q,
                    vbias=vbias,
                    U=U,
                    num_visibles=num_visibles,
                )
                all_train_ll.append(new_ll.item())
                all_test_ll.append(new_test_ll.item())
                grad_vbias_norm = torch.norm(grad_vbias_pos - grad_vbias_neg)
                log_string = build_log_string_train(
                    train_ll=new_ll.item(),
                    test_ll=new_test_ll.item(),
                    n_iter=n_iter,
                    mean_q=q.mean().item(),
                    grad_vbias_norm=grad_vbias_norm.item(),
                    curr_lr=learning_rate,
                    count=count_lr,
                )
                count_lr = 0
                pbar.write(log_string)
                if torch.abs(prev_test_ll - new_test_ll) < stop_ll:
                    break
                prev_test_ll = prev_test_ll * 0.05 + new_test_ll * 0.95
    total_iter += n_iter
    all_train_ll = torch.tensor(all_train_ll)
    all_test_ll = torch.tensor(all_test_ll)
    return best_vbias, best_q, all_train_ll, all_test_ll, total_iter
