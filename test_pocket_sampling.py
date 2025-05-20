import torch
import argparse
from pathlib import Path
import os
import sys
import time

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

# We'll test the main parts of the sampling process to see where it hangs
from qm9.sampling import sample
from qm9.utils import prepare_context_pocket
from equivariant_diffusion.utils import assert_mean_zero_with_mask


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


class DummyArgs:
    def __init__(self):
        self.context_node_nf = 64
        self.conditioning = ["pocket"]
        self.probabilistic_model = "diffusion"
        self.include_charges = True


def test_pocket_sampling():
    print("Testing pocket sampling...")

    # Create test environment
    device = torch.device("cpu")
    args = DummyArgs()
    dataset_info = {"max_n_nodes": 19, "atom_decoder": ["H", "C", "N", "O", "F"]}

    # Create dummy data
    batch_size = 2
    max_n_nodes = dataset_info["max_n_nodes"]
    nodesxsample = torch.tensor([10, 15])

    # Create node mask
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0 : nodesxsample[i]] = 1
    node_mask = node_mask.unsqueeze(2).to(device)

    # Create pocket encoding
    pocket_encoding = torch.randn(batch_size, args.context_node_nf).to(device)

    print(f"Initial test values:")
    print(f"  batch_size: {batch_size}")
    print(f"  max_n_nodes: {max_n_nodes}")
    print(f"  node_mask shape: {node_mask.shape}")
    print(f"  pocket_encoding shape: {pocket_encoding.shape}")

    print("\nTesting prepare_context_pocket...")
    dummy_positions = torch.zeros(batch_size, max_n_nodes, 3).to(device)
    atom_mask = node_mask.squeeze(-1)
    context = prepare_context_pocket(pocket_encoding, dummy_positions, atom_mask)
    print(f"prepare_context_pocket output shape: {context.shape}")

    print("\nTesting sample function...")
    try:
        model = DummyModel()
        one_hot, charges, x, node_mask_out = sample(
            args,
            device,
            model,
            dataset_info,
            None,
            nodesxsample=nodesxsample,
            context=pocket_encoding,
        )

        print("\nSampling successful!")
        print(f"one_hot shape: {one_hot.shape}")
        print(f"charges shape: {charges.shape}")
        print(f"x shape: {x.shape}")
        print(f"node_mask shape: {node_mask_out.shape}")

    except Exception as e:
        print(f"\nError during sampling: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pocket_sampling()
