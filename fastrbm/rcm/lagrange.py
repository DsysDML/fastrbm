import torch
from rbms.custom_fn import log2cosh
from torch import Tensor
from tqdm import tqdm


def get_configurational_entropy(m: Tensor, mu: Tensor, U: Tensor) -> Tensor:
    """
    Calculate the configurational entropy.

    Parameters
    ----------
    m : Tensor
        The point in the longitudinal space. (n_dim,)
    mu : Tensor
        The Lagrange multiplier associated to m. (n_dim,)
    U : Tensor
        The projection matrix of shape. (n_dim, n_visible)

    Returns
    -------
    Tensor
        The configurational entropy. (1,)
    """
    n_vis = U.shape[1]
    sqrn = n_vis**0.5

    # Compute phi[mu]
    h = mu @ U * sqrn
    configurational_entropy = log2cosh(h).mean() - m @ mu

    return configurational_entropy


def vec_compute_grad_phi(th: Tensor, U: Tensor) -> Tensor:
    """
    Compute the gradient of the phi function with respect to the input tensor.

    Parameters:
    ----------
    th : Tensor
        The input tensor. (batch_size, n_dim)
    U : Tensor
        The weight tensor. (n_dim, n_visible)

    Returns:
    ----------
    Tensor
        The gradient of the phi function with respect to the input tensor. (batch_size, num_visibles).
    """
    num_visibles = U.shape[1]
    return torch.mm(U, th.t()).T / torch.sqrt(torch.tensor(num_visibles))


def vec_compute_hessian_phi(var: Tensor, U: Tensor) -> Tensor:
    """
    Compute the Hessian matrix of the phi function.

    Parameters:
    ----------
    var : Tensor
        The input tensor. (batch_size, n_visible)
    U : Tensor
        The weight tensor. (n_dim, n_visible)

    Returns:
    ----------
    Tensor
        The Hessian matrix of the phi function. (batch, n_dim, n_dim)
    """
    # Add a singleton dimension to U to make it compatible with var
    U = U.unsqueeze(0)
    # Vectorized U diag(var) U.T with the same U matrix and a batch of var
    return torch.einsum("lij, bj, ljk -> bik ", U, var, U.permute(0, 2, 1))


def get_lagrange_multipliers(
    m: Tensor,
    U: Tensor,
    err_threshold: float = 1e-10,
    max_iter: int = 100,
):
    """
    Calculate the Lagrange multipliers.

    Parameters
    ----------
    m : Tensor
        The points in the longitudinal space. (num_points, intrinsic_dimension)
    U : Tensor
        The projection matrix. (intrinsic_dimension, n_visible)
    err_threshold : float, optional
        The error threshold for convergence, by default 1e-10.
    max_iter : int, optional
        The maximum number of iterations, by default 100.
    num_colors: int
        Dummy arg to have the same signature as get_lagrange_multipliers_potts

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor]
        A tuple containing the following:
        - done_m : Tensor
            The points that have converged. (num_converged_points, intrinsic_dimension)
        - done_mu : Tensor
            The Lagrange multipliers associated to the converged points. (num_converged_points, intrinsic_dimension)
        - configurational_entropy : Tensor
            The configurational entropy for each converged point. (num_converged_points,)
    """
    num_points, intrinsic_dimension = m.shape
    n_visible = U.shape[1]
    sqrn = torch.sqrt(torch.tensor([n_visible], device=m.device))
    err = 10

    mu = torch.zeros((num_points, intrinsic_dimension), device=m.device, dtype=m.dtype)
    done_m = []
    done_mu = []
    with tqdm(range(max_iter), leave=False) as pbar:
        pbar.set_description("Computing Lagrange multipliers")
        for n_iter in pbar:
            th = torch.tanh(sqrn * mu @ U)
            to_keep = torch.linalg.vector_norm(th, dim=1, ord=float("inf")) < 1
            m = m[to_keep]
            mu = mu[to_keep]
            th = th[to_keep]
            var = 1 - torch.pow(th, 2)
            grad_phi = vec_compute_grad_phi(th, U)
            H = vec_compute_hessian_phi(var, U)
            dmu = torch.linalg.solve(H, (m - grad_phi))
            mu += dmu
            err = torch.linalg.vector_norm(dmu, ord=float("inf"), dim=1)
            to_store = err <= err_threshold
            if to_store.sum() > 0:
                done_m.append(m[to_store])
                done_mu.append(mu[to_store])
                m = m[torch.logical_not(to_store)]
                mu = mu[torch.logical_not(to_store)]
            if len(m) == 0:
                break
    if len(done_m) > 0:
        done_m = torch.vstack(done_m)
        done_mu = torch.vstack(done_mu)
    else:
        done_m = torch.tensor([], device=m.device, dtype=m.dtype)
        done_mu = torch.tensor([], device=m.device, dtype=m.dtype)

    num_converged_points = done_m.shape[0]
    configurational_entropy = torch.zeros(
        num_converged_points, device=done_m.device, dtype=done_m.dtype
    )
    for i in range(num_converged_points):
        configurational_entropy[i] = get_configurational_entropy(
            done_m[i], done_mu[i], U
        )
    return (done_m, done_mu, configurational_entropy)
