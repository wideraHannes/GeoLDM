"""
Test script to verify that the protein pocket context embeddings are correctly generated
and passed through the data pipeline.
"""

import sys
from pathlib import Path
import torch
import traceback

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src.geoldm.my_ext.crossdock_dataset import get_dataloaders
    print("Successfully imported crossdock_dataset")
except ImportError as e:
    print(f"Error importing get_dataloaders: {e}")
    print(traceback.format_exc())
    sys.exit(1)


def main():
    # Set up paths and parameters
    root_dir = "crossdocked/crossdocked_pocket10"
    batch_size = 2
    subset = 4  # Small subset for testing

    # Load a small batch of data
    loaders = get_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        subset=subset,
        max_ligand_atoms=128,
        max_pocket_atoms=512,
    )

    # Get a batch from the training data
    batch = next(iter(loaders["train"]))

    # Print batch information
    print("\nBatch keys:", batch.keys())

    # Check the context embeddings
    print("\nContext embedding (global):")
    print("Shape:", batch["context"].shape)
    print("Type:", batch["context"].dtype)
    print("First few values:", batch["context"][0, :5])

    # Check the per-node context features
    print("\nContext node features (per-atom):")
    print("Shape:", batch["context_node_features"].shape)
    print("Type:", batch["context_node_features"].dtype)
    print("First atom's context:", batch["context_node_features"][0, 0, :5])
    
    # Verify that all atoms in a molecule share the same context
    first_mol_context = batch["context_node_features"][0, 0]
    second_atom_context = batch["context_node_features"][0, 1]
    are_equal = torch.allclose(first_mol_context, second_atom_context)
    print(f"\nAll atoms in same molecule share context: {are_equal}")
    
    # Verify that different molecules have different contexts
    if batch_size > 1:
        first_mol_context = batch["context_node_features"][0, 0]
        second_mol_context = batch["context_node_features"][1, 0]
        are_different = not torch.allclose(first_mol_context, second_mol_context)
        print(f"Different molecules have different contexts: {are_different}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
