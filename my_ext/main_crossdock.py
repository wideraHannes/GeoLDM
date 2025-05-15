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
    from my_ext.ESM_pocket_encoder import ESM2PocketEncoder

    encoder = ESM2PocketEncoder()
    import train_test

    orig_train_epoch = train_test.train_epoch
    orig_test = train_test.test

    def extract_pocket_context(batch):
        pdb_paths = batch["pdb_path"]  # (B,) list of strings
        pocket_codes = []
        for i, pdb_path in enumerate(pdb_paths):
            pocket_vec = encoder.encode_pdb(Path(pdb_path))  # (64,)
            pocket_codes.append(pocket_vec)
        context = torch.stack(pocket_codes).to(batch["positions"].device)  # (B,64)
        context = context.unsqueeze(1).expand(-1, batch["positions"].shape[1], -1)
        return context

    def train_epoch_with_context(*args, **kwargs):
        loader = kwargs.get("loader", args[1] if len(args) > 1 else None)
        for batch in loader:
            context = extract_pocket_context(batch)
            batch["context"] = context
        return orig_train_epoch(*args, **kwargs)

    def test_with_context(*args, **kwargs):
        loader = kwargs.get("loader", args[1] if len(args) > 1 else None)
        for batch in loader:
            context = extract_pocket_context(batch)
            batch["context"] = context
        return orig_test(*args, **kwargs)

    train_test.train_epoch = train_epoch_with_context
    train_test.test = test_with_context

    # Call main as usual
    from main_qm9 import main

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
        "--n_report_steps",
        "5",
    ]
    run_with_pocket_context()
