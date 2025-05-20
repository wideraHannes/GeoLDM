import sys
import torch
import argparse
from pathlib import Path
import os
import time

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from qm9.sampling import sample
from qm9.utils import prepare_context_pocket


class DummyModel:
    def sample(self, batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=False):
        print(f"MODEL SAMPLE: Starting with batch_size={batch_size}, max_n_nodes={max_n_nodes}")
        print(f"MODEL SAMPLE: node_mask shape={node_mask.shape}, edge_mask shape={edge_mask.shape}")
        print(f"MODEL SAMPLE: context shape={context.shape if context is not None else None}")

        # Simulate generation by adding sleep
        print("MODEL SAMPLE: Starting diffusion process...")
        for i in range(5):
            print(f"MODEL SAMPLE: Step {i + 1}/5")
            time.sleep(0.1)  # Sleep for 0.1 second to simulate computation

        print("MODEL SAMPLE: Diffusion complete, returning dummy tensors")
        # Return dummy tensors
        device = node_mask.device
        x = torch.zeros(batch_size, max_n_nodes, 3).to(device)  # positions
        h = {
            "categorical": torch.zeros(batch_size, max_n_nodes, 5).to(device),  # one-hot
            "integer": torch.zeros(batch_size, max_n_nodes, 1).to(device),  # charges
        }
        return x, h


class TestArgs:
    def __init__(self):
        self.context_node_nf = 64
        self.conditioning = ["pocket"]
        self.probabilistic_model = "diffusion"
        self.include_charges = True


def generate_molecules_with_pocket():
    print("Testing molecule generation with pocket conditioning...")

    # Set up environment
    device = torch.device("cpu")
    args = TestArgs()

    # Create a simple pocket encoding (would normally come from ESM model)
    batch_size = 2
    pocket_encoding = torch.randn(batch_size, args.context_node_nf).to(device)

    # Create a dataset info dictionary
    dataset_info = {
        "max_n_nodes": 19,
        "atom_decoder": ["H", "C", "N", "O", "F"],
    }

    # Setup dummy model
    model = DummyModel()

    # Setup nodesxsample (number of nodes per sample)
    nodesxsample = torch.tensor([10, 15])

    # Generate samples
    print("\nStarting molecule generation...")
    one_hot, charges, positions, node_mask = sample(
        args, device, model, dataset_info, None, nodesxsample=nodesxsample, context=pocket_encoding
    )

    print("\nGeneration successful!")
    print(f"Generated molecules:")
    print(f"  Batch size: {batch_size}")
    print(f"  Positions shape: {positions.shape}")
    print(f"  One-hot shape: {one_hot.shape}")
    print(f"  Charges shape: {charges.shape}")


if __name__ == "__main__":
    generate_molecules_with_pocket()

if __name__ == "__main__":
    generate_molecules_with_pocket()
