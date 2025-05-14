import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from test_dataset_pocket import CrossDockedPoseDataset, collate_fn, get_dataloaders
from qm9.models import get_latent_diffusion
from configs.datasets_config import get_dataset_info
import pickle
from os.path import join


class PocketEncoder(nn.Module):
    """
    Encoder for processing pocket information.
    This module takes pocket coordinates and features and produces a pocket encoding
    that can be used to condition the ligand generation.
    """

    def __init__(self, hidden_dim=128, output_dim=64):
        super().__init__()
        self.pocket_encoder = nn.Sequential(
            nn.Linear(3 + 28 + 1, hidden_dim),  # 3D coords + atom types + charge
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pocket_positions, pocket_features, pocket_mask):
        # Process pocket information
        pocket_input = torch.cat([pocket_positions, pocket_features], dim=-1)
        pocket_encoding = self.pocket_encoder(pocket_input)
        # Mask out padding
        pocket_encoding = pocket_encoding * pocket_mask
        # Pool to get fixed-size representation
        pocket_encoding = pocket_encoding.sum(dim=1) / (pocket_mask.sum(dim=1) + 1e-8)
        return pocket_encoding


class PocketConditionedGeoLDM(nn.Module):
    """
    Main model for pocket-conditioned ligand generation.
    This extends the original GeoLDM by incorporating pocket information
    as a conditioning signal.
    """

    def __init__(self, pretrained_model_path, device="cuda"):
        super().__init__()
        self.device = device

        # Load pretrained GeoLDM
        with open(join(pretrained_model_path, "args.pickle"), "rb") as f:
            self.args = pickle.load(f)
        self.args.cuda = False
        if hasattr(self.args, "device"):
            self.args.device = "cpu"

        # Create base model
        self.base_model, self.nodes_dist, self.prop_dist = get_latent_diffusion(
            self.args, device, get_dataset_info("qm9", remove_h=False), None
        )

        # Load pretrained weights
        self.base_model.load_state_dict(
            torch.load(
                join(pretrained_model_path, "generative_model_ema.npy"),
                map_location=device,
            )
        )

        # Add pocket encoder
        self.pocket_encoder = PocketEncoder().to(device)

        # Add conditioning layers
        self.conditioning_layer = nn.Sequential(
            nn.Linear(64, 128),  # 64 is pocket encoding dim
            nn.ReLU(),
            nn.Linear(128, self.args.latent_nf),  # Match latent dim of base model
        ).to(device)

    def prepare_input(self, batch):
        """
        Prepare input tensors from batch data.
        Separates ligand and pocket information.
        """
        # Extract data
        positions = batch["positions"].to(self.device)
        one_hot = batch["one_hot"].to(self.device)
        charges = batch["charges"].to(self.device)
        atom_mask = batch["atom_mask"].to(self.device)
        pocket_mask = batch["pocket_mask"].to(self.device)
        edge_mask = batch["edge_mask"].to(self.device).float()

        # Separate ligand and pocket
        ligand_mask = atom_mask.bool()
        pocket_mask = pocket_mask.bool()

        # Get pocket information
        pocket_positions = positions * pocket_mask.unsqueeze(-1)
        pocket_features = torch.cat(
            [one_hot, charges.unsqueeze(-1)], dim=-1
        ) * pocket_mask.unsqueeze(-1)

        # Get ligand information
        ligand_positions = positions * ligand_mask.unsqueeze(-1)
        ligand_features = torch.cat(
            [one_hot, charges.unsqueeze(-1)], dim=-1
        ) * ligand_mask.unsqueeze(-1)

        return {
            "ligand_positions": ligand_positions,
            "ligand_features": ligand_features,
            "ligand_mask": ligand_mask,
            "pocket_positions": pocket_positions,
            "pocket_features": pocket_features,
            "pocket_mask": pocket_mask,
            "edge_mask": edge_mask,
        }

    def forward(self, batch):
        """
        Forward pass for training.
        Processes pocket information and conditions the diffusion process.
        """
        # Prepare input
        data = self.prepare_input(batch)

        # Encode pocket information
        pocket_encoding = self.pocket_encoder(
            data["pocket_positions"], data["pocket_features"], data["pocket_mask"]
        )

        # Get conditioning signal
        conditioning = self.conditioning_layer(pocket_encoding)

        # Run diffusion process with conditioning
        loss = self.base_model(
            data["ligand_positions"],
            data["ligand_features"],
            data["ligand_mask"],
            data["edge_mask"],
            context=conditioning,  # Pass pocket conditioning
        )

        return loss

    def sample(self, pocket_batch, n_samples=1):
        """
        Generate ligands conditioned on pocket information.
        """
        # Prepare pocket information
        data = self.prepare_input(pocket_batch)

        # Encode pocket
        pocket_encoding = self.pocket_encoder(
            data["pocket_positions"], data["pocket_features"], data["pocket_mask"]
        )

        # Get conditioning
        conditioning = self.conditioning_layer(pocket_encoding)

        # Generate ligands
        with torch.no_grad():
            output_x, output_h = self.base_model.sample(
                n_samples=n_samples,
                n_nodes=data["ligand_positions"].shape[1],
                node_mask=data["ligand_mask"],
                edge_mask=data["edge_mask"],
                context=conditioning,
            )

        return output_x, output_h


def train_model():
    """
    Training loop for the pocket-conditioned model.
    """
    # Initialize model
    model = PocketConditionedGeoLDM(
        pretrained_model_path="outputs/qm9_latent2",
        device="cpu",  # Force CPU usage
    )

    # Initialize dataloaders using the function from test_dataset_pocket
    loaders = get_dataloaders(
        root_dir="crossdocked/crossdocked_pocket10",
        batch_size=4,  # Reduced batch size for CPU
        num_workers=0,  # No parallel workers for CPU
        max_ligand_atoms=128,
        max_pocket_atoms=512,
        radius=6.0,
        remove_h=True,
    )

    # Get the training dataloader
    dataloader = loaders["train"]
    val_dataloader = loaders["val"]

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(100):
        total_loss = 0
        for batch in dataloader:
            # Forward pass
            loss = model(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    loss = model(batch)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
            model.train()


if __name__ == "__main__":
    train_model()
