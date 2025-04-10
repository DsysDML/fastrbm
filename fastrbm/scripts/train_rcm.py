import argparse

from rbms.dataset import load_dataset
from rbms.dataset.parser import add_args_dataset
from rbms.parser import add_args_pytorch, remove_argument

from fastrbm.rcm.parser import add_args_rcm
from fastrbm.rcm.train import train


def create_parser():
    parser = argparse.ArgumentParser("Train a Restricted Coulomb Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rcm(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "variable_type")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args = vars(args)
    args["variable_type"] = "Ising"

    dataset, _ = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=1.0,
        binarize=True,
    )
    print(dataset)
    dataset.data = dataset.data * 2 - 1

    train(
        dataset=dataset,
        args=args,
        mesh_file=args["mesh_file"],
    )
