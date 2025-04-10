import argparse
from pathlib import Path


def add_args_rcm(parser: argparse.ArgumentParser):
    rcm_args = parser.add_argument_group("RCM")
    parser.add_argument(
        "-o",
        "--filename",
        type=Path,
        default="RCM.h5",
        help="(Defaults to RCM.h5). Path to the file where to save the model.",
    )
    rcm_args.add_argument(
        "--mesh_file",
        type=str,
        required=True,
        help="Path to a precomputed mesh.",
    )
    rcm_args.add_argument(
        "--save_all_trial",
        default=False,
        help="(Defaults to False). Save all trial of RCM. Useful when decimation is True.",
        action="store_true",
    )
    rcm_args.add_argument(
        "--num_hidden",
        type=int,
        default=100,
        help="(Defaults to 100). Number of hidden units.",
    )
    rcm_args.add_argument(
        "--num_sample_gen",
        type=int,
        default=2_000,
        help="(Defaults to 2 000). Number of sample to generate post-training",
    )
    rcm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rcm_args.add_argument(
        "--max_iter",
        type=int,
        default=100_000,
        help="(Defaults to 100 000). Maximum number of iteration per epoch.",
    )
    rcm_args.add_argument(
        "--smooth_rate",
        type=float,
        default=0.1,
        help="(Defaults to 0.1). Smoothing rate for the update of the hessian.",
    )
    rcm_args.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
        help="(Defaults to 1e-6). r_min.",
    )
    rcm_args.add_argument(
        "--adapt",
        default=False,
        help="(Defaults to False). Use an adaptive learning rate strategy.",
        action="store_true",
    )
    rcm_args.add_argument(
        "--stop_ll",
        type=float,
        default=1e-2,
        help="(Defaults to 1e-2). Log-likelihood precision for early stopping.",
    )
    rcm_args.add_argument(
        "--feature_threshold",
        type=int,
        default=500,
        help="(Defaults to 500). Feature threshold for feature decimation.",
    )
    rcm_args.add_argument(
        "--eigen_threshold",
        type=float,
        default=1e-4,
        help="(Defaults to 1e-4). Minimum eigenvalue for the hessian.",
    )
    rcm_args.add_argument(
        "--decimation",
        default=False,
        help="(Defaults to False). Decimate features.",
        action="store_true",
    )
    rcm_args.add_argument(
        "--seed_rcm",
        type=int,
        default=8127394031293,
        help="(Defaults to None). Seed for the RCM method. Does not change the seed for dataset splitting.",
    )
    rcm_args.add_argument(
        "--max_num_hiddens",
        default=None,
        type=int,
        help="(Defaults to None). Upper bound on the final number of hidden nodes.",
    )
    return parser
