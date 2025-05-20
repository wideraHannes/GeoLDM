import sys
import torch
import argparse
from pathlib import Path
import os
import numpy as np
import traceback

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import after setting up sys.path
from qm9.sampling import sample
from qm9.utils import prepare_context_pocket

class DummyArgs:
    def __init__(self):
        self.conditioning = ["pocket"]
        self.context_node_nf = 64  # Must match the pocket encoding dimension
        self.probabilistic_model = "diffusion"
        self.include_charges = True

def test_sampling():
    print("Testing sampling with pocket conditioning...")
    
    # Create dummy arguments
    args = DummyArgs()
    
    # Create a simple device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a dummy model that returns predefined tensors
    class DummyModel:
        def sample(self, batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=False):
            print(f"DummyModel.sample called with context shape: {context.shape if context is not None else None}")
            # Return dummy tensors with the expected shapes
            x = torch.zeros(batch_size, max_n_nodes, 3).to(device)  # positions
            h = {
                "categorical": torch.zeros(batch_size, max_n_nodes, 5).to(device),  # one-hot atom types
                "integer": torch.zeros(batch_size, max_n_nodes, 1).to(device)  # charges
            }
            return x, h
    
    # Create a dummy dataset info
    dataset_info = {
        "max_n_nodes": 19,
        "atom_decoder": ["H", "C", "N", "O", "F"]  # 5 atom types
    }
    
    # Test with a dummy pocket encoding
    batch_size = 2
    max_n_nodes = dataset_info["max_n_nodes"]
    
    # Create a dummy pocket encoding
    pocket_encoding = torch.randn(batch_size, args.context_node_nf).to(device)
    print(f"Created pocket_encoding with shape: {pocket_encoding.shape}")
    
    # Create dummy node mask
    nodesxsample = torch.tensor([10, 15])  # Different number of nodes per sample
    node_mask = torch.zeros(batch_size, max_n_nodes)
    for i in range(batch_size):
        node_mask[i, 0:nodesxsample[i]] = 1
    node_mask = node_mask.unsqueeze(2).to(device)
    print(f"Created node_mask with shape: {node_mask.shape}")
    
    # Create dummy positions
    dummy_positions = torch.zeros(batch_size, max_n_nodes, 3).to(device)
    
    try:
        # Test the prepare_context_pocket function separately
        print("Testing prepare_context_pocket function...")
        context = prepare_context_pocket(pocket_encoding, dummy_positions, node_mask.squeeze(-1))
        print(f"Context shape after prepare_context_pocket: {context.shape}")
        
        # Don't actually run the sample function yet, just print dimensions
        print("\nSetting up sample function call...")
        print(f"args.context_node_nf: {args.context_node_nf}")
        print(f"args.conditioning: {args.conditioning}")
        print(f"nodesxsample: {nodesxsample}")
        print(f"pocket_encoding shape: {pocket_encoding.shape}")
        
        print("\nNow calling sample function with pocket_encoding...")
        # NOTE: We're intentionally passing the 2D encoding to test our fix
        one_hot, charges, x, node_mask_out = sample(
            args, device, DummyModel(), dataset_info, None, 
            nodesxsample=nodesxsample, context=pocket_encoding
        )
        
        print("\nSampling successful!")
        print(f"one_hot shape: {one_hot.shape}")
        print(f"charges shape: {charges.shape}")
        print(f"x shape: {x.shape}")
        print(f"node_mask shape: {node_mask_out.shape}")
        
    except Exception as e:
        print(f"\nError during sampling: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_sampling()
