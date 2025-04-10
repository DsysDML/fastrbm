import argparse
import pathlib
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from rbms.bernoulli_bernoulli.classes import BBRBM
from rbms.const import LOG_FILE_HEADER

from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.io import save_model
from rbms.parser import add_args_pytorch
from rbms.partition_function.ais import (
    compute_partition_function_ais,
)
from rbms.partition_function.exact import compute_partition_function
from rbms.sampling.gibbs import sample_state
from rbms.utils import compute_log_likelihood, get_categorical_configurations

from fastrbm.rcm.rbm import sample_rbm


def add_args_convert(parser: argparse.ArgumentParser):
    convert_args = parser.add_argument_group("Convert")
    convert_args.add_argument(
        "--path",
        "-i",
        type=Path,
        required=True,
        help="Path to the folder h5 archive of the RCM.",
    )
    convert_args.add_argument(
        "--output",
        "-o",
        type=Path,
        default="RBM.h5",
        help="(Defaults to RBM.h5). Path to the file where to save the model in RBM format.",
    )
    convert_args.add_argument(
        "--num_hiddens",
        type=int,
        default=50,
        help="(Defaults to 50). Target number of hidden nodes for the RBM.",
    )
    convert_args.add_argument(
        "--therm_steps",
        type=int,
        default=10000,
        help="(Defaults to 1e4). Number of steps to be performed to thermalize the chains.",
    )
    convert_args.add_argument(
        "--trial",
        type=int,
        default=None,
        help="(Defaults to the best trial). RCM trial to use",
    )

    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rbm_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=20,
        help="(Defaults to 10). Number of Gibbs steps for each gradient estimation.",
    )
    rbm_args.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="(Defaults to 1000). Minibatch size.",
    )
    rbm_args.add_argument(
        "--seed_rcm",
        type=int,
        default=945723295,
        help="(Defaults to 9457232957489). Seed for the experiments.",
    )
    rbm_args.add_argument(
        "--num_chains",
        default=2000,
        type=int,
        help="(Defaults to 2000). The number of permanent chains.",
    )
    rbm_args.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="(Defaults to False). Create a log file.",
    )
    return parser


def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert RCM into an RBM readable format."
    )
    parser = add_args_dataset(parser)
    parser = add_args_convert(parser)
    parser = add_args_pytorch(parser)
    return parser


def ising_to_bernoulli(params: BBRBM) -> BBRBM:
    params.vbias = 2.0 * (params.vbias - params.weight_matrix.sum(1))
    params.hbias = 2.0 * (-params.hbias - params.weight_matrix.sum(0))
    params.weight_matrix = 4.0 * params.weight_matrix
    return params


def convert(args: dict, device: torch.device, dtype: torch.dtype):
    dataset, _ = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=1.0,
        device=args["device"],
        dtype=args["dtype"],
    )
    # Set the random seed
    if args["seed_rcm"] is not None:
        np.random.seed(args["seed_rcm"])
        torch.manual_seed(args["seed_rcm"])

    if args["trial"] is None:
        trial_name = "best_trial"
    else:
        trial_name = f"trial_{args['trial']}"

    # Import parameters

    print(f"Trial selected: {trial_name}")
    with h5py.File(args["path"], "r") as f:
        args["seed"] = f["hyperparameters"]["seed"][()].item()
        args["train_size"] = f["hyperparameters"]["train_size"][()].item()
        vbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["vbias_rbm"])).to(device).to(dtype)
        )
        hbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["hbias_rbm"])).to(device).to(dtype)
        )
        weight_matrix_rcm = (
            torch.from_numpy(np.array(f[trial_name]["W_rbm"])).to(device).to(dtype)
        )
        parallel_chains_v = (
            torch.from_numpy(np.array(f[trial_name]["samples_gen"]))
            .to(device)
            .to(dtype)
        )
        p_m = torch.from_numpy(np.array(f[trial_name]["pdm"])).to(device).to(dtype)
        m = torch.from_numpy(np.array(f["const"]["m"])).to(device).to(dtype)
        mu = torch.from_numpy(np.array(f["const"]["mu"])).to(device).to(dtype)
        U = torch.from_numpy(np.array(f["const"]["U"])).to(device).to(dtype)
        if "time" in f.keys():
            total_time = np.array(f["time"]).item()
        else:
            total_time = 0

    rng = np.random.default_rng(args["seed"])
    train_dataset, test_dataset = dataset.split_train_test(
        rng, train_size=args["train_size"]
    )

    start = time.time()
    params = BBRBM(weight_matrix=weight_matrix_rcm, vbias=vbias_rcm, hbias=hbias_rcm)
    params = ising_to_bernoulli(params=params)
    num_visibles, num_hiddens_rcm = params.weight_matrix.shape

    num_hiddens_add = args["num_hiddens"] - num_hiddens_rcm
    if num_hiddens_add < 0:
        print("The target number of hidden nodes is lower than the RCMs one.")
        num_hiddens_add = 0
    print(f"Adding {num_hiddens_add} hidden nodes.")

    hbias_add = torch.zeros(size=(num_hiddens_add,), device=device)
    weight_matrix_add = (
        torch.randn(size=(num_visibles, num_hiddens_add), device=device) * 1e-4
    )
    params.hbias = torch.cat([params.hbias, hbias_add])
    params.weight_matrix = torch.cat([params.weight_matrix, weight_matrix_add], dim=1)
    num_hiddens = num_hiddens_rcm + num_hiddens_add

    parallel_chains_v = sample_rbm(p_m, mu, U, args["num_chains"], device, dtype)
    # Convert parallel chains into (0, 1) format
    parallel_chains_v = (parallel_chains_v + 1) / 2

    # Thermalize chains
    print("Thermalizing the parallel chains...")
    num_chains = len(parallel_chains_v)

    parallel_chains = params.init_chains(
        num_samples=num_chains, start_v=parallel_chains_v
    )
    parallel_chains = sample_state(
        gibbs_steps=args["therm_steps"], chains=parallel_chains, params=params
    )

    # Compute initial log partition function
    if min(params.weight_matrix.shape[0], params.weight_matrix.shape[-1]) <= 20:
        all_config = get_categorical_configurations(
            n_states=2,
            n_dim=min(params.weight_matrix.shape),
            device=device,
            dtype=dtype,
        )

        log_z = compute_partition_function(params=params, all_config=all_config)
    else:
        log_z = compute_partition_function_ais(
            num_chains=1000, num_beta=5000, params=params
        )
    log_weights = torch.zeros(
        parallel_chains["visible"].shape[0], device=device, dtype=dtype
    )

    train_ll = compute_log_likelihood(
        v_data=train_dataset.data,
        w_data=train_dataset.weights,
        params=params,
        log_z=log_z,
    )
    test_ll = compute_log_likelihood(
        v_data=test_dataset.data,
        w_data=test_dataset.weights,
        params=params,
        log_z=log_z,
    )

    # Index at which to save the current model
    curr_update = 1

    with h5py.File(args["output"], "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_chains"] = num_chains
        hyperparameters["batch_size"] = args["batch_size"]
        hyperparameters["gibbs_steps"] = args["gibbs_steps"]
        hyperparameters["filename"] = str(args["output"])
        hyperparameters["learning_rate"] = args["learning_rate"]
        hyperparameters["seed"] = args["seed"]
        hyperparameters["train_size"] = args["train_size"]
        rcm = file_model.create_group("rcm")
        rcm["U"] = U.cpu().numpy()
        rcm["mu"] = mu.cpu().numpy()
        rcm["m"] = m.cpu().numpy()
        rcm["pdm"] = p_m.cpu().numpy()

    save_model(
        filename=args["output"],
        params=params,
        chains=parallel_chains,
        num_updates=1,
        time=time.time() - start + total_time,
        flags=["ptt"],
    )

    # Tr-AIS parameters
    with h5py.File(args["output"], "a") as file_model:
        file_model[f"update_{curr_update}"]["log_z"] = log_z
        file_model[f"update_{curr_update}"]["log_weights"] = log_weights.cpu().numpy()
        file_model[f"update_{curr_update}"]["train_ll"] = train_ll
        file_model[f"update_{curr_update}"]["test_ll"] = test_ll

    if args["log"]:
        filename = pathlib.Path(args["output"])
        log_filename = filename.parent / pathlib.Path(f"log-{filename.stem}.csv")
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write(",".join(LOG_FILE_HEADER) + "\n")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    convert(args, args["device"], args["dtype"])
