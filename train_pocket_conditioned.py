import torch
from torch.utils.data import DataLoader
from pocket_dataset import ProteinPocketDataset
from qm9.models import get_latent_diffusion
from configs.datasets_config import get_dataset_info
from os.path import join
import pickle
import wandb
from tqdm import tqdm
import os


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        # Prepare input
        positions = batch["positions"].to(device)  # [B, N, 3]
        one_hot = batch["one_hot"].to(device)  # [B, N, num_atom_types]
        charges = batch["charges"].to(device)  # [B, N]
        atom_mask = batch["atom_mask"].to(device)  # [B, N1]
        pocket_mask = batch["pocket_mask"].to(device)  # [B, N2]
        edge_mask = batch["edge_mask"].to(device).float()  # [B, N, N]

        if i == 0:
            print("positions shape:", positions.shape)
            print("atom_mask shape:", atom_mask.shape)
            print("pocket_mask shape:", pocket_mask.shape)
            print("one_hot shape:", one_hot.shape)
            print("charges shape:", charges.shape)
            print("edge_mask shape:", edge_mask.shape)

        # Pad atom_mask and pocket_mask if needed
        N = positions.shape[1]
        if atom_mask.shape[1] != N:
            pad = N - atom_mask.shape[1]
            atom_mask = torch.nn.functional.pad(atom_mask, (0, pad), value=0)
        if pocket_mask.shape[1] != N:
            pad = N - pocket_mask.shape[1]
            pocket_mask = torch.nn.functional.pad(pocket_mask, (0, pad), value=0)

        # Prepare node_mask for masking (combine atom and pocket masks)
        node_mask = (atom_mask.bool() | pocket_mask.bool()).float()
        if node_mask.dim() == 2:
            node_mask = node_mask.unsqueeze(-1)  # [B, N, 1]

        # Mask positions and features
        positions = positions * node_mask
        one_hot = one_hot * node_mask
        charges = charges * node_mask

        # Combine features into a single tensor
        h = torch.cat([one_hot, charges], dim=-1)  # [B, N, num_atom_types + 1]

        # Debug: check masking before model call
        if i == 0:
            print(
                "Max abs(positions * (1 - node_mask)):",
                (positions * (1 - node_mask)).abs().max().item(),
            )
            print(
                "Max abs(h * (1 - node_mask)):",
                (h * (1 - node_mask)).abs().max().item(),
            )

        # Center positions using all atoms in node_mask
        node_mask_f = node_mask.float()
        masked_positions = positions * node_mask_f
        num_atoms = node_mask_f.sum(dim=1, keepdim=True)
        mean = masked_positions.sum(dim=1, keepdim=True) / (num_atoms + 1e-8)
        positions = positions - mean * node_mask_f
        positions = positions * node_mask_f
        # Debug: verify mean is zero for node_mask atoms
        if i == 0:
            masked_positions = positions * node_mask_f
            mean_after = masked_positions.sum(dim=1, keepdim=True) / (num_atoms + 1e-8)
            print("Mean after centering (node_mask):", mean_after.abs().max().item())

        # Forward pass
        loss = model(positions, h, node_mask, edge_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            positions = batch["positions"].to(device)
            one_hot = batch["one_hot"].to(device)
            charges = batch["charges"].to(device)
            atom_mask = batch["atom_mask"].to(device)
            pocket_mask = batch["pocket_mask"].to(device)
            edge_mask = batch["edge_mask"].to(device).float()

            if i == 0:
                print("[VAL] positions shape:", positions.shape)
                print("[VAL] atom_mask shape:", atom_mask.shape)
                print("[VAL] pocket_mask shape:", pocket_mask.shape)
                print("[VAL] one_hot shape:", one_hot.shape)
                print("[VAL] charges shape:", charges.shape)
                print("[VAL] edge_mask shape:", edge_mask.shape)

            N = positions.shape[1]
            if atom_mask.shape[1] != N:
                pad = N - atom_mask.shape[1]
                atom_mask = torch.nn.functional.pad(atom_mask, (0, pad), value=0)
            if pocket_mask.shape[1] != N:
                pad = N - pocket_mask.shape[1]
                pocket_mask = torch.nn.functional.pad(pocket_mask, (0, pad), value=0)

            h = {"categorical": one_hot, "integer": charges}

            # Prepare node_mask for masking (combine atom and pocket masks)
            node_mask = (atom_mask.bool() | pocket_mask.bool()).float()
            if node_mask.dim() == 2:
                node_mask = node_mask.unsqueeze(-1)  # [B, N, 1]

            # Center positions using all atoms in node_mask
            node_mask_f = node_mask.float()
            masked_positions = positions * node_mask_f
            num_atoms = node_mask_f.sum(dim=1, keepdim=True)
            mean = masked_positions.sum(dim=1, keepdim=True) / (num_atoms + 1e-8)
            positions = positions - mean * node_mask_f
            positions = positions * node_mask_f
            # Debug: verify mean is zero for node_mask atoms
            if i == 0:
                masked_positions = positions * node_mask_f
                mean_after = masked_positions.sum(dim=1, keepdim=True) / (
                    num_atoms + 1e-8
                )
                print(
                    "[VAL] Mean after centering (node_mask):",
                    mean_after.abs().max().item(),
                )

            loss = model(positions, h, node_mask, edge_mask)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    # Initialize wandb
    wandb.init(
        project="pocket-conditioned-geoldm",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 100,
            "max_ligand_atoms": 128,
            "max_pocket_atoms": 512,
            "radius": 6.0,
        },
    )

    # Set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset info
    dataset_info = get_dataset_info("qm9", remove_h=False)

    # Load pretrained model
    model_path = "outputs/qm9_latent2"
    print(f"Loading model from: {model_path}")
    with open(join(model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    # Force CPU usage in args
    args.cuda = False

    # Create model
    model, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, None
    )

    # Load pretrained weights
    model.load_state_dict(
        torch.load(join(model_path, "generative_model_ema.npy"), map_location="cpu")
    )
    model.to(device)

    # Patch encoder and decoder embedding layers for new input feature size (29)
    new_in_node_nf = 29  # 28 atom types + 1 charge
    # Patch encoder embedding
    old_linear = model.vae.encoder.egnn.embedding
    hidden_nf = old_linear.out_features
    new_linear = torch.nn.Linear(new_in_node_nf, hidden_nf)
    with torch.no_grad():
        n_old = min(old_linear.in_features, new_in_node_nf)
        new_linear.weight[:, :n_old] = old_linear.weight[:, :n_old]
        new_linear.bias = old_linear.bias
    model.vae.encoder.egnn.embedding = new_linear
    # Patch decoder embedding
    old_linear_dec = model.vae.decoder.egnn.embedding
    hidden_nf_dec = old_linear_dec.out_features
    new_linear_dec = torch.nn.Linear(new_in_node_nf, hidden_nf_dec)
    with torch.no_grad():
        n_old = min(old_linear_dec.in_features, new_in_node_nf)
        new_linear_dec.weight[:, :n_old] = old_linear_dec.weight[:, :n_old]
        new_linear_dec.bias = old_linear_dec.bias
    model.vae.decoder.egnn.embedding = new_linear_dec

    # Initialize datasets
    train_dataset = ProteinPocketDataset(
        root_dir="crossdocked/crossdocked_pocket10",
        split="train",
        max_ligand_atoms=wandb.config.max_ligand_atoms,
        max_pocket_atoms=wandb.config.max_pocket_atoms,
        radius=wandb.config.radius,
        remove_h=True,
    )

    val_dataset = ProteinPocketDataset(
        root_dir="crossdocked/crossdocked_pocket10",
        split="val",
        max_ligand_atoms=wandb.config.max_ligand_atoms,
        max_pocket_atoms=wandb.config.max_pocket_atoms,
        radius=wandb.config.radius,
        remove_h=True,
    )

    # Create dataloaders with reduced batch size for CPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,  # Reduced batch size for CPU
        shuffle=True,
        num_workers=0,  # No parallel workers for CPU
        pin_memory=False,  # No pin memory for CPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,  # Reduced batch size for CPU
        shuffle=False,
        num_workers=0,  # No parallel workers for CPU
        pin_memory=False,  # No pin memory for CPU
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    # Create output directory
    os.makedirs("outputs/pocket_conditioned", exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(wandb.config.epochs):
        print(f"\nEpoch {epoch + 1}/{wandb.config.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, device)

        # Log metrics
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), join("outputs/pocket_conditioned", "best_model.pt")
            )

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            join("outputs/pocket_conditioned", f"checkpoint_epoch_{epoch}.pt"),
        )


if __name__ == "__main__":
    main()
