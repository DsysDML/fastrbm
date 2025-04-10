# Fast training and sampling of Restricted Boltzmann Machines

For all the scripts, add `-h` at the end to get an explanations of all command line arguments.

## Installation

1. Clone this repository

```bash
git clone https://github.com/DsysDML/fastrbm
```

2. Install the repository

```bash
cd fastrbm && pip install .
```

## Compute the mesh for the RCM

```bash
rcm mesh -d path/to/data.h5 --subset_labels 0 1  --dimension 0 1 2 \
--with_bias -o path/to/output.h5
```

## Train the RCM

```bash
rcm train -d path/to/data.h5 --mesh_file path/to/mesh.h5 --num_hidden 100 \
--adapt --decimation --filename path/to/output.h5
```

## Map the RCM to a RBM

```bash
rcm to_rbm -d path/to/data.h5 -i path/to/rcm.h5 -o path/to/output.h5 \
--num_hiddens 200 --therm_steps 1000 --gibbs_steps 100 --batch_size 2000 \
--num_chains 2000 --learning_rate 0.01
```

## Restore the training from a RBM

```bash
fastrbm train -d path/to/data.h5 --filename path/to/rbm.h5  \
--num_updates 10000 --restore
```

## Train a RBM from scratch

```bash
fastrbm train -d path/to/data.h5 --filename path/to/rbm.h5 \
--num_updates 10000 --learning_rate 0.01 --batch_size 2000 \
--num_chains 2000 --gibbs_steps 100
```

## Analyze a posteriori

See [this notebook](notebook/analyse_rbm.ipynb)

## Cite this work

```
@inproceedings{bereux2025fast,
  title={Fast training and sampling of Restricted Boltzmann Machines},
  author={B{\'e}reux, Nicolas and Decelle, Aur{\'e}lien and Furtlehner, Cyril and Rosset, Lorenzo and Seoane, Beatriz},
  booktitle={13th International Conference on Learning Representations-ICLR 2025},
  year={2025}
}
```
