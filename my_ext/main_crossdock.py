import sys
import torch
import argparse
from pathlib import Path
import os


# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Add the qm9 directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "qm9"))

from qm9 import dataset as qm9_dataset
import configs.datasets_config as dsc
from my_ext.crossdock_info import crossdock_pocket10
from my_ext.crossdock_dataset import CrossDockedPoseDataset, collate_fn

print("cwd:", os.getcwd())

# ----- 1.  Monkey-patch dataset info -----

_get = dsc.get_dataset_info
dsc.get_dataset_info = (
    lambda name, rm_h: crossdock_pocket10 if name == "crossdock_pocket10" else _get(name, rm_h)
)

# ----- 2.  Monkey-patch dataset factory -----


def get_dataset(args, _):
    train = CrossDockedPoseDataset(args.dataset_path, split="train")
    val = CrossDockedPoseDataset(args.dataset_path, split="val")
    test = CrossDockedPoseDataset(args.dataset_path, split="test")
    return train, val, test


qm9_dataset.get_dataset = get_dataset

# ----- 3.  CLI defaults cloned from main_qm9.py -----


# Patch retrieve_dataloaders to support crossdock_pocket10
def patched_retrieve_dataloaders(cfg):
    if cfg.dataset == "crossdock_pocket10":
        from my_ext.crossdock_dataset import get_dataloaders

        dataloaders = get_dataloaders(
            root_dir=cfg.datadir,
            batch_size=cfg.batch_size,
            num_workers=0,  # Force single-process loading for ESM compatibility
        )
        charge_scale = None
        return dataloaders, charge_scale
    else:
        return original_retrieve_dataloaders(cfg)


import qm9.dataset

original_retrieve_dataloaders = qm9.dataset.retrieve_dataloaders
qm9.dataset.retrieve_dataloaders = patched_retrieve_dataloaders


def run_with_pocket_context():
    from main_qm9 import main, args

    if getattr(args, "dataset", None) == "crossdock_pocket10":
        args.context_node_nf = 64
    main()


if __name__ == "__main__":
    sys.argv += [
        "--dataset",
        "crossdock_pocket10",
        "--datadir",
        "./crossdocked/crossdocked_small/",
        "--n_epochs",
        "10",
        "--batch_size",
        "2",
        "--latent_nf",
        "1",
        "--exp_name",
        "poc_crossdock",
        "--train_diffusion",  # keep VAE frozen!
        "--test_epochs",
        "1",
        "--no-cuda",
        "--conditioning",
        "pocket",
        "--diffusion_steps",
        "1",
    ]
    run_with_pocket_context()
