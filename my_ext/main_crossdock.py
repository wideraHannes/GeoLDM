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
from main_qm9 import main
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

if __name__ == "__main__":
    sys.argv += [
        "--dataset",
        "crossdock_pocket10",
        "--dataset_path",
        "./crossdocked/crossdocked_pocket10",
        "--n_epochs",
        "75",
        "--batch_size",
        "8",
        "--latent_nf",
        "1",
        "--exp_name",
        "poc_crossdock",
        "--train_diffusion",  # keep VAE frozen!
    ]
    main()
