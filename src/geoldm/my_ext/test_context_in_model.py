"""
This script demonstrates how to use the protein pocket context embeddings in a training loop.
"""

import sys
import torch
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.geoldm.my_ext.crossdock_dataset import get_dataloaders


class SimpleNetwork(torch.nn.Module):
    """
    A simple network that takes atom features and context and predicts atom types.
    This is just for demonstration purposes.
    """
    
    def __init__(self, atom_dim, context_dim, hidden_dim=128):
        super().__init__()
        self.atom_encoder = torch.nn.Linear(atom_dim, hidden_dim)
        self.context_encoder = torch.nn.Linear(context_dim, hidden_dim)
        self.combined = torch.nn.Linear(hidden_dim * 2, atom_dim)
        
    def forward(self, atom_features, context_features):
        # Encode atom features
        atom_hidden = torch.relu(self.atom_encoder(atom_features))
        
        # Encode context features
        context_hidden = torch.relu(self.context_encoder(context_features))
        
        # Combine features
        combined = torch.cat([atom_hidden, context_hidden], dim=-1)
        
        # Predict atom types
        output = self.combined(combined)
        
        return output


def main():
    print("Starting main function")
    # Set up paths and parameters
    root_dir = "crossdocked/crossdocked_pocket10"
    batch_size = 2
    subset = 4  # Small subset for testing
    
    print("Loading dataloaders...")
    # Load a small batch of data
    try:
        loaders = get_dataloaders(
            root_dir=root_dir,
            batch_size=batch_size,
            subset=subset,
            max_ligand_atoms=128,
            max_pocket_atoms=512,
        )
        print("Dataloaders loaded successfully")
    except Exception as e:
        print(f"Error loading dataloaders: {e}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Initialize the model
    atom_dim = 5  # Number of atom types
    context_dim = 64  # Context dimension
    model = SimpleNetwork(atom_dim, context_dim)
    
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Get a batch from the training data
    batch = next(iter(loaders["train"]))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Extract relevant tensors from batch
    atom_features = batch["one_hot"]  # (B, N, atom_dim)
    context_features = batch["context_node_features"]  # (B, N, context_dim)
    
    # Select only atoms that are part of the ligand (based on atom_mask)
    atom_mask = batch["atom_mask"].bool()
    
    # Print shapes before model
    print(f"Atom features shape: {atom_features.shape}")
    print(f"Context features shape: {context_features.shape}")
    print(f"Atom mask shape: {atom_mask.shape}")
    
    # Forward pass
    pred = model(atom_features, context_features)
    
    # Calculate loss (reconstruction loss on atom features)
    loss = torch.nn.functional.mse_loss(pred, atom_features)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item()}")
    print("Gradient check - atom_encoder weight grad norm:", 
          torch.norm(model.atom_encoder.weight.grad).item())
    print("Gradient check - context_encoder weight grad norm:", 
          torch.norm(model.context_encoder.weight.grad).item())
    
    if model.context_encoder.weight.grad is not None:
        print("Context features are being used in the model!")
    else:
        print("WARNING: Context features are not being used!")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
