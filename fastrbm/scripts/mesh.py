import argparse

import h5py
import numpy as np
import torch
from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.parser import add_args_pytorch, remove_argument

from fastrbm.rcm.mesh import (
    batched_lagrange_mult,
    compute_mesh,
    entropy_correction,
)
from fastrbm.rcm.pca import compute_U
from rbms.parser import set_args_default, default_args


def create_parser():
    parser = argparse.ArgumentParser(
        "Discretize the space along a projection of the dataset."
    )
    parser = add_args_dataset(parser)
    parser = add_args_pytorch(parser)
    parser.add_argument(
        "--dimension",
        nargs="+",
        help="The dimensions on which to do RCM",
        required=True,
    )
    parser.add_argument(
        "--n_pts_dim",
        type=int,
        default=100,
        help="(Defaults to 100). The number of discretization point on each direction. The mesh will have n_pts_dim**n_dim points.",
    )
    parser.add_argument(
        "-o",
        "--filename",
        type=str,
        default="mesh.h5",
        help="(Defaults to mesh.h5). Name of the file to write the mesh to.",
    )
    parser.add_argument(
        "--with_bias",
        default=False,
        action="store_true",
        help="(Defaults to False). Center the dataset and use the resulting bias as the first direction.",
    )
    parser.add_argument(
        "--num_colors",
        type=int,
        default=21,
        help="(Defaults to 21). Number of possible states for Potts variable.",
    )
    remove_argument(parser, "variable_type")
    remove_argument(parser, "dtype")
    return parser


def main(args: dict):
    dtype = torch.float64
    train_dataset, _ = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        device=args["device"],
        binarize=True,
        dtype=dtype,
    )
    if args["data"][-5:] == "fasta":
        train_dataset.is_binary = False
    intrinsic_dimension = len(args["dimension"])
    train_set = train_dataset.data * 2 - 1
    print(train_dataset)
    U, bias = compute_U(
        M=train_set,
        weights=train_dataset.weights,
        intrinsic_dimension=intrinsic_dimension,
        device=args["device"],
        dtype=dtype,
        with_bias=args["with_bias"],
    )
    U = U.T

    # We do a first iteration on the unit ball to get the limits of the domain
    n_pts_dim = 50
    bias_lim = None
    proj_data = train_set @ U.T / U.shape[1] ** 0.5
    if args["with_bias"]:
        bias_lim = torch.tensor(
            [proj_data.min(0).values[0], proj_data.max(0).values[0]]
        )
    # bias_lim = None
    m = compute_mesh(
        U=U,
        n_pts_dim=n_pts_dim,
        device=args["device"],
        dtype=dtype,
        with_bias=args["with_bias"],
        bias=bias,
        bias_lim=bias_lim,
    )

    m, _, _ = batched_lagrange_mult(
        m=m,
        U=U,
        mesh_desc="coarse-grained",
        num_colors=args["num_colors"],
    )

    torch.cuda.empty_cache()

    # We do it again with a better discretization
    border_length = 0.04  # 2/50 i.e. the distance to the next not valid points
    n_pts_dim = args["n_pts_dim"]
    dim_min = m.min(0).values - border_length
    dim_max = m.max(0).values + border_length
    m = compute_mesh(
        U=U,
        n_pts_dim=n_pts_dim,
        device=args["device"],
        dtype=dtype,
        with_bias=args["with_bias"],
        bias=bias,
        dim_min=dim_min,
        dim_max=dim_max,
        bias_lim=bias_lim,
    )

    m, mu, configurational_entropy = batched_lagrange_mult(
        m=m,
        U=U,
        mesh_desc="fine-grained",
        num_colors=args["num_colors"],
    )
    torch.cuda.empty_cache()

    configurational_entropy = entropy_correction(
        m=m, mu=mu, configurational_entropy=configurational_entropy, U=U
    )
    # Save the results
    with h5py.File(args["filename"], "w") as f:
        f["m"] = m.cpu().numpy()
        f["mu"] = mu.cpu().numpy()
        f["configurational_entropy"] = configurational_entropy.cpu().numpy()
        f["U"] = U.cpu().numpy()
        hyperparameters = f.create_group("hyperparameters")
        hyperparameters["with_bias"] = args["with_bias"]
        hyperparameters["n_pts_dim"] = args["n_pts_dim"]
        hyperparameters["dimension"] = np.array([int(d) for d in args["dimension"]])
        hyperparameters["seed"] = args["seed"]
        hyperparameters["train_size"] = args["train_size"]


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    args = set_args_default(args, default_args)
    args["variable_type"] = "Ising"
    args["dtype"] = "double"
    main(args)
