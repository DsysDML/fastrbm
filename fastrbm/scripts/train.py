import argparse
import time

import numpy as np
import torch
from rbms.dataset import load_dataset
from rbms.dataset.dataset_class import RBMDataset
from rbms.dataset.parser import add_args_dataset
from rbms.io import save_model
from rbms.map_model import map_model
from rbms.parser import (
    add_args_pytorch,
    add_args_rbm,
    add_args_saves,
    default_args,
    remove_argument,
    set_args_default,
)
from rbms.partition_function.ais import update_weights_ais
from rbms.potts_bernoulli.classes import PBRBM
from rbms.potts_bernoulli.utils import (
    ensure_zero_sum_gauge,
)
from rbms.sampling.gibbs import sample_state
from rbms.training.pcd import fit_batch_pcd
from rbms.training.utils import create_machine, get_checkpoints, setup_training
from rbms.utils import check_file_existence, compute_log_likelihood, log_to_csv
from torch.optim import SGD

from fastrbm.io import load_rcm
from fastrbm.trajectory.ais import (
    init_ais_traj_params,
    load_ais_traj_params,
    save_ais_traj_params,
)
from fastrbm.trajectory.pt import ptt_sampling, swap_config_multi
from fastrbm.utils import clone_dict, compute_ess


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rbm(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    parser.add_argument(
        "--acc_with_ptt",
        default=False,
        action="store_true",
        help="(Defaults to False). Perform 1 PTT step to estimate acceptance rate at every step.",
    )
    remove_argument(parser, "use_torch")
    return parser


def train(
    dataset: RBMDataset,
    model_type: str,
    args: dict,
    dtype: torch.dtype,
    checkpoints: np.ndarray,
):
    """Train the Potts-Bernoulli RBM model.

    Args:
        dataset (RBMDataset): The training dataset.
        test_dataset (RBMDataset): The test dataset (not used).
        args (dict): A dictionary of training arguments.
        dtype (torch.dtype): The data type for the parameters.
        checkpoints (np.ndarray): An array of checkpoints for saving model states.
    """

    filename = args["filename"]
    num_visibles = dataset.get_num_visibles()

    if not (args["overwrite"]):
        check_file_existence(filename)

    num_visibles = dataset.get_num_visibles()
    # Create a first archive with the initialized model
    if not (args["restore"]):
        args = set_args_default(args=args, default_args=default_args)
        rng = np.random.default_rng(args["seed"])
        train_dataset, test_dataset = dataset.split_train_test(
            rng, args["train_size"], args["test_size"]
        )
        params = map_model[model_type].init_parameters(
            num_hiddens=args["num_hiddens"],
            dataset=dataset,
            device=args["device"],
            dtype=dtype,
        )
        create_machine(
            filename=args["filename"],
            params=params,
            num_visibles=num_visibles,
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            learning_rate=args["learning_rate"],
            train_size=args["train_size"],
            log=args["log"],
            flags=["ptt", "checkpoint"],
            seed=args["seed"],
        )
        init_ais_traj_params(
            filename=args["filename"],
            batch_size=args["batch_size"],
            device=args["device"],
            dtype=dtype,
            map_model=map_model,
        )
    (
        params,
        parallel_chains,
        args,
        num_updates,
        start,
        elapsed_time,
        log_filename,
        pbar,
        train_dataset,
        test_dataset,
    ) = setup_training(args, map_model=map_model, dataset=dataset)
    args = set_args_default(args=args, default_args=default_args)

    # Load AIS trajectory parameters
    log_z_init, log_weights = load_ais_traj_params(args["filename"], num_updates)
    log_weights = torch.from_numpy(log_weights).to(device=args["device"], dtype=dtype)

    curr_params = params.clone()

    # Define the optimizer
    optimizer = SGD(params.parameters(), lr=args["learning_rate"], maximize=True)

    # Save PTT
    list_params_save_ptt = [params.clone()]
    list_chains_save_ptt = [clone_dict(parallel_chains)]

    # Initialize gradients for the parameters
    for p in params.parameters():
        p.grad = torch.zeros_like(p)

    rcm = load_rcm(args["filename"], device=args["device"], dtype=dtype)

    log_z_init_ais = log_z_init

    chains_ll_ais = params.init_chains(parallel_chains["visible"].shape[0])
    chains_ll_ais = sample_state(100, chains_ll_ais, params)

    with torch.no_grad():
        for idx in range(num_updates + 1, args["num_updates"] + 1):
            flags = []
            prev_params = curr_params.clone()
            curr_params = params.clone()
            log_weights, chains_ll_ais = update_weights_ais(
                prev_params,
                curr_params,
                chains_ll_ais,
                log_weights=log_weights,
                n_steps=1,
            )

            rand_idx = torch.randperm(len(train_dataset))[: args["batch_size"]]
            batch = (train_dataset.data[rand_idx], train_dataset.weights[rand_idx])
            optimizer.zero_grad(set_to_none=False)
            parallel_chains, logs = fit_batch_pcd(
                batch=batch,
                parallel_chains=parallel_chains,
                params=params,
                gibbs_steps=args["gibbs_steps"],
                beta=args["beta"],
                centered=True,
            )

            # Beginning Tr-AIS LL
            log_z_ais = (
                torch.logsumexp(log_weights, 0)
                - np.log(parallel_chains["visible"].shape[0])
                + log_z_init_ais
            ).item()
            ess = compute_ess(log_weights)
            if ess < 0.3:
                from fastrbm.utils import systematic_resampling

                # chains_ll_ais = clone_dict(parallel_chains)
                chains_ll_ais = systematic_resampling(chains_ll_ais, log_weights)
                log_z_init_ais = log_z_ais
                log_weights = torch.zeros_like(log_weights)
            train_ll_ais = compute_log_likelihood(
                v_data=dataset.data,
                w_data=dataset.weights,
                params=params,
                log_z=log_z_ais,
            )
            test_ll_ais = compute_log_likelihood(
                v_data=test_dataset.data,
                w_data=test_dataset.weights,
                params=params,
                log_z=log_z_ais,
            )

            pbar.set_postfix_str(
                f"train LL AIS: {np.mean(train_ll_ais):.2f}, test LL AIS: {np.mean(test_ll_ais):.2f}"
            )
            # End Tr-AIS LL

            # Apply gradient update
            optimizer.step()
            if isinstance(params, PBRBM):
                ensure_zero_sum_gauge(params)

            compute_acc_rate_ptt = True
            target_acc_rate_ptt = args["target_acc_rate"]
            if compute_acc_rate_ptt:
                # Permute
                rand_idx = torch.randperm(list_chains_save_ptt[-1]["visible"].shape[0])
                for k, v in list_chains_save_ptt[-1].items():
                    list_chains_save_ptt[-1][k] = v[rand_idx]
                if args["acc_with_ptt"]:
                    list_chains_save_ptt, acc_rate_ptt, _ = ptt_sampling(
                        [*list_params_save_ptt, params],
                        [*list_chains_save_ptt, parallel_chains],
                        rcm,
                        1,
                        1,
                        False,
                        False,
                    )
                    list_chains_save_ptt.pop(-1)
                else:
                    _, acc_rate_ptt, _ = swap_config_multi(
                        params=[list_params_save_ptt[-1], params],
                        chains=[list_chains_save_ptt[-1], parallel_chains],
                        index=None,
                        perform_swap=False,
                    )

                save_ptt = acc_rate_ptt[-1] < target_acc_rate_ptt
                if save_ptt:
                    pbar.write(f"{acc_rate_ptt}")
                    list_params_save_ptt.append(prev_params.clone())
                    list_chains_save_ptt.append(clone_dict(parallel_chains))
                    flags.append("ptt")
            if idx in checkpoints:
                flags.append("checkpoint")
            if idx == args["num_updates"] and "ptt" not in flags:
                flags.append("ptt")
            # We save only when there is a flag
            if len(flags) > 0:
                curr_time = time.time() - start
                save_model(
                    filename=args["filename"],
                    params=params,
                    chains=parallel_chains,
                    num_updates=idx,
                    time=curr_time + elapsed_time,
                    flags=flags,
                )
                save_ais_traj_params(
                    filename=args["filename"],
                    age=idx,
                    log_z=log_z_ais,
                    log_weights=log_weights.cpu().numpy(),
                )
            if args["log"]:
                log_to_csv(logs, log_file=log_filename)

            # Update progress bar
            pbar.update(1)


def train_rbm(args: dict):
    checkpoints = get_checkpoints(
        num_updates=args["num_updates"], n_save=args["n_save"], spacing=args["spacing"]
    )
    dataset, _ = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        binarize=args["binarize"],
        train_size=1.0,
        test_size=None,
        device=args["device"],
        dtype=args["dtype"],
    )
    print(dataset)
    if dataset.is_binary:
        model_type = "BBRBM"
    else:
        model_type = "PBRBM"
    train(
        dataset=dataset,
        model_type=model_type,
        args=args,
        dtype=args["dtype"],
        checkpoints=checkpoints,
    )


def main():
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    parser.add_argument("--target_acc_rate", type=float, default=0.3)
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    train_rbm(args=args)


if __name__ == "__main__":
    main()
