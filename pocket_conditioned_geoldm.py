import torch
from torch.utils.data import DataLoader
from pocket_dataset import ProteinPocketDataset
from qm9.models import get_latent_diffusion
from configs.datasets_config import get_dataset_info
from os.path import join
import pickle
from qm9 import visualizer as vis


class PocketConditionedGeoLDM:
    def __init__(self, model_path, device="cpu"):
        # Load dataset info
        self.dataset_info = get_dataset_info("qm9", remove_h=False)

        # Load pretrained model
        print(f"Loading model from: {model_path}")
        with open(join(model_path, "args.pickle"), "rb") as f:
            self.args = pickle.load(f)

        # Force CPU usage in args
        self.args.cuda = False
        self.device = torch.device(device)

        # Create model
        self.model, self.nodes_dist, self.prop_dist = get_latent_diffusion(
            self.args, self.device, self.dataset_info, None
        )

        # Load pretrained weights
        self.model.load_state_dict(
            torch.load(join(model_path, "generative_model_ema.npy"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Initialize dataset
        self.dataset = ProteinPocketDataset(
            root_dir="crossdocked/crossdocked_pocket10",
            split="train",
            max_ligand_atoms=128,
            max_pocket_atoms=512,
            radius=6.0,
            remove_h=True,
        )

        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)

    def prepare_pocket_conditioned_input(self, batch):
        """Prepare input with pocket information for the model."""
        # Extract ligand and pocket information
        positions = batch["positions"][0]  # [N, 3]
        one_hot = batch["one_hot"][0]  # [N, num_atom_types]
        charges = batch["charges"][0]  # [N]
        atom_mask = batch["atom_mask"][0]  # [N]
        pocket_mask = batch["pocket_mask"][0]  # [N]
        edge_mask = batch["edge_mask"][0]  # [N, N]

        # Move to device
        positions = positions.to(self.device)
        one_hot = one_hot.to(self.device)
        charges = charges.to(self.device)
        atom_mask = atom_mask.to(self.device).float().unsqueeze(-1)
        pocket_mask = pocket_mask.to(self.device).float().unsqueeze(-1)
        edge_mask = edge_mask.to(self.device).float()

        # Combine ligand and pocket features
        h = {"categorical": one_hot, "integer": charges}

        # Center positions using only ligand atoms
        atom_mask.squeeze(-1).bool()
        masked_positions = positions * atom_mask
        num_ligand_atoms = atom_mask.sum()
        mean = masked_positions.sum(dim=0) / (num_ligand_atoms + 1e-8)
        positions = positions - mean
        positions = positions * (atom_mask + pocket_mask)  # Keep both ligand and pocket

        return positions, h, atom_mask, pocket_mask, edge_mask

    def generate_ligand(self, pocket_batch):
        """Generate a ligand conditioned on the pocket."""
        # Prepare input
        x, h, node_mask, pocket_mask, edge_mask = self.prepare_pocket_conditioned_input(
            pocket_batch
        )

        # Sample from the model
        with torch.no_grad():
            output_x, output_h = self.model.sample(
                n_samples=x.shape[0],
                n_nodes=x.shape[1],
                node_mask=node_mask,
                edge_mask=edge_mask,
                context=None,  # We'll add pocket conditioning here in future versions
            )

        return output_x, output_h

    def visualize_generation(self, output_x, output_h):
        """Visualize the generated ligand."""
        vis.plot_data3d(
            output_x.squeeze(0).cpu(),
            torch.argmax(output_h["categorical"].squeeze(0).cpu(), dim=1),
            self.dataset_info,
            spheres_3d=True,
        )


def main():
    # Initialize model
    model = PocketConditionedGeoLDM(model_path="outputs/qm9_latent2", device="cpu")

    # Get a sample from the dataset
    batch = next(iter(model.dataloader))

    # Generate ligand
    output_x, output_h = model.generate_ligand(batch)

    # Visualize
    model.visualize_generation(output_x, output_h)


if __name__ == "__main__":
    main()
