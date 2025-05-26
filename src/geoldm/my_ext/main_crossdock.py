from src.geoldm.qm9 import dataset as qm9_dataset
import src.geoldm.configs.datasets_config as dsc
from src.geoldm.my_ext.crossdock_info import crossdock_pocket10
from src.geoldm.my_ext.crossdock_dataset import CrossDockedPoseDataset
import sys
from configs.paths import PRETRAINED_QM9

# ----- 1.  Monkey-patch dataset info -----

""" _get = dsc.get_dataset_info
dsc.get_dataset_info = (
    lambda name, rm_h: crossdock_pocket10 if name == "crossdock_pocket10" else _get(name, rm_h)
) """

# ----- 2.  Monkey-patch dataset factory -----


""" def get_dataset(args, _):
    train = CrossDockedPoseDataset(args.dataset_path, split="train")
    val = CrossDockedPoseDataset(args.dataset_path, split="val")
    test = CrossDockedPoseDataset(args.dataset_path, split="test")
    return train, val, test


qm9_dataset.get_dataset = get_dataset """

# ----- 3.  CLI defaults cloned from main_qm9.py -----

""" 
# Patch retrieve_dataloaders to support crossdock_pocket10
def patched_retrieve_dataloaders(cfg):
    if cfg.dataset == "crossdock_pocket10":
        from src.geoldm.my_ext.crossdock_dataset import get_dataloaders

        dataloaders = get_dataloaders(
            root_dir=cfg.datadir,
            batch_size=cfg.batch_size,
            num_workers=0,  # Force single-process loading for ESM compatibility
        )
        charge_scale = None
        return dataloaders, charge_scale
    else:
        return original_retrieve_dataloaders(cfg)
 """

""" from src.geoldm import qm9

original_retrieve_dataloaders = qm9.dataset.retrieve_dataloaders
qm9.dataset.retrieve_dataloaders = patched_retrieve_dataloaders """


def main():
    from scripts.main_qm9 import main, args

    main()


if __name__ == "__main__":
    sys.argv += [
        "--dataset",
        "crossdock_pocket10",
        "--datadir",
        "./crossdocked/crossdocked_5atoms_small/",
        "--n_epochs",
        "30",
        "--batch_size",
        "4",
        "--exp_name",
        "poc_crossdock",
        # "--train_diffusion",  # keep VAE frozen!
        "--test_epochs",
        "5",
        "--no-cuda",
        "--exp_name",
        "poc_crossdock_debug",  # New experiment name
        # "--conditioning", # remove conditioning for now
        # "pocket",
        "--resume",
        str(PRETRAINED_QM9),
        "--diffusion_steps",
        "1000",
        "--latent_nf",  # default is 4,
        "2",
        "--lr",
        "0.001",
        "--train_diffusion",
        # "--trainable_ae",
        # "--context_node_nf",
        # "64"
    ]
    main()
