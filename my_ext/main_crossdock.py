import sys
import torch
import argparse
from pathlib import Path
import os
import time
import copy


# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Add the qm9 directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "qm9"))

from qm9 import dataset as qm9_dataset
import configs.datasets_config as dsc
from my_ext.crossdock_info import crossdock_pocket10
from my_ext.crossdock_dataset import CrossDockedPoseDataset, collate_fn
from my_ext.pocket_encoder import PocketEncoder
from qm9.models import get_optim, get_autoencoder, get_latent_diffusion
from equivariant_diffusion import utils as flow_utils
import utils

print("cwd:", os.getcwd())

# ----- 1.  Monkey-patch dataset info -----

_get = dsc.get_dataset_info
dsc.get_dataset_info = (
    lambda name, rm_h: crossdock_pocket10 if name == "crossdock_pocket10" else _get(name, rm_h)
)

# ----- 2.  Custom dataloader function -----


def retrieve_dataloaders(args):
    print(f"Creating custom CrossDockedPoseDataset with path: {args.dataset_path}")

    train_dataset = CrossDockedPoseDataset(args.dataset_path, split="train")
    val_dataset = CrossDockedPoseDataset(args.dataset_path, split="val")
    test_dataset = CrossDockedPoseDataset(args.dataset_path, split="test")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    dataloaders = {"train": train_loader, "valid": val_loader, "test": test_loader}

    charge_scale = 1.0  # Default value

    return dataloaders, charge_scale


# ----- 3. Create pocket encoder and attach to the training process -----


class PocketContextProvider:
    def __init__(self, device):
        self.pocket_encoder = PocketEncoder().to(device)

    def get_context(self, batch):
        pocket_pos = batch["positions"] * batch.get("pocket_mask", None)
        pocket_feat = torch.cat([batch["x"], batch["charges"]], dim=-1)
        pocket_feat = pocket_feat * batch.get("pocket_mask", None)
        pocket_mask = batch.get("pocket_mask", None)

        if pocket_mask is None:
            return None

        # Encode pocket features
        return self.pocket_encoder(pocket_pos, pocket_feat, pocket_mask)


# ----- 4. Modified training with pocket context -----


def train_with_pocket_context(args, dataloaders, device, dtype):
    dataset_info = dsc.get_dataset_info(args.dataset, args.remove_h)

    # Initialize pocket context provider
    pocket_context_provider = PocketContextProvider(device)

    # Create model
    if args.train_diffusion:
        print("Setting up latent diffusion model...")
        model, nodes_dist, prop_dist = get_latent_diffusion(
            args, device, dataset_info, dataloaders["train"]
        )
    else:
        print("Setting up autoencoder model...")
        model, nodes_dist, prop_dist = get_autoencoder(
            args, device, dataset_info, dataloaders["train"]
        )

    model = model.to(device)
    optim = get_optim(args, model)

    # Initialize dataparallel if enabled
    if args.dp and torch.cuda.device_count() > 1:
        print(f"Training using {torch.cuda.device_count()} GPUs")
        model_dp = torch.nn.DataParallel(model.cpu())
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    # Initialize EMA model
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.DataParallel(model_ema)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    print("Model initialized and ready for training.")
    print(f"Starting training with {args.n_epochs} epochs...")

    # Just print that we're about to start training and return
    print("Model and training setup complete. First epoch would start now.")
    return model


# ----- 5.  Command-line interface -----

if __name__ == "__main__":
    # Parse args with all required parameters
    parser = argparse.ArgumentParser(description="GeoLDM with Pocket Context")

    # Add essential arguments
    parser.add_argument("--dataset", type=str, default="crossdock_pocket10")
    parser.add_argument("--dataset_path", type=str, default="./crossdocked/crossdocked_pocket10")
    parser.add_argument("--n_epochs", type=int, default=75)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_nf", type=int, default=1)
    parser.add_argument("--diffusion_steps", type=int, default=500)
    parser.add_argument("--exp_name", type=str, default="pocket_conditioned_geoldm")
    parser.add_argument("--train_diffusion", action="store_true", default=True)
    parser.add_argument("--ae_path", type=str, default=None, help="Path to pretrained autoencoder")
    parser.add_argument("--trainable_ae", action="store_true", default=False)
    parser.add_argument("--dp", action="store_true", default=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--nf", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--test_epochs", type=int, default=10)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--remove_h", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--context_node_nf", type=int, default=0)
    parser.add_argument("--diffusion_noise_schedule", type=str, default="polynomial_2")
    parser.add_argument("--diffusion_noise_precision", type=float, default=1e-5)
    parser.add_argument("--diffusion_loss_type", type=str, default="l2")
    parser.add_argument("--no-cuda", action="store_true", default=False)

    args = parser.parse_args()

    # Setup device
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")

    # Setup output folders without wandb
    os.makedirs(f"outputs/{args.exp_name}", exist_ok=True)

    try:
        # Get dataloaders using our custom function
        print("Retrieving dataloaders...")
        dataloaders, charge_scale = retrieve_dataloaders(args)
        print(f"Dataloaders created successfully")

        # Train with pocket context - just setup, don't start training
        model = train_with_pocket_context(args, dataloaders, device, dtype)

        print("Setup completed successfully!")

    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback

        traceback.print_exc()
