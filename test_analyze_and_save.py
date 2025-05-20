import sys
import torch
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from train_test import analyze_and_save
from qm9.utils import prepare_context_pocket
import traceback

class MockModel:
    def __init__(self, device):
        self.device = device
    
    def sample(self, batch_size, max_n_nodes, node_mask, edge_mask, context, fix_noise=False):
        print(f"MockModel.sample called with:")
        print(f"  batch_size: {batch_size}")
        print(f"  max_n_nodes: {max_n_nodes}")
        print(f"  node_mask shape: {node_mask.shape}")
        print(f"  edge_mask shape: {edge_mask.shape}")
        print(f"  context shape: {context.shape if context is not None else None}")
        
        # Create dummy results
        x = torch.zeros(batch_size, max_n_nodes, 3).to(self.device)  # positions
        one_hot = torch.zeros(batch_size, max_n_nodes, 5).to(self.device)  # atom types
        charges = torch.zeros(batch_size, max_n_nodes, 1).to(self.device)  # charges
        
        return x, {"categorical": one_hot, "integer": charges}

class MockNodesDist:
    def sample(self, batch_size):
        # Return a tensor with random number of nodes between 5 and 15
        return torch.randint(5, 16, (batch_size,))

class MockArgs:
    def __init__(self):
        self.batch_size = 4
        self.context_node_nf = 64
        self.conditioning = ["pocket"]
        self.probabilistic_model = "diffusion"
        self.include_charges = True

def test_analyze_and_save():
    print("Testing analyze_and_save with pocket conditioning...")
    
    # Set up mock objects
    device = torch.device("cpu")
    model = MockModel(device)
    nodes_dist = MockNodesDist()
    args = MockArgs()
    
    # Create a mock dataset_info
    dataset_info = {
        "max_n_nodes": 19,
        "atom_decoder": ["H", "C", "N", "O", "F"],
        "name": "qm9"
    }
    
    # Create a mock property distribution (not used with pocket conditioning)
    prop_dist = None
    
    try:
        print("Calling analyze_and_save...")
        result = analyze_and_save(
            epoch=1,
            model_sample=model,
            nodes_dist=nodes_dist,
            args=args,
            device=device,
            dataset_info=dataset_info,
            prop_dist=prop_dist,
            n_samples=2
        )
        print("analyze_and_save completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Error during analyze_and_save: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_analyze_and_save()
