import torch
from torch.utils.data import DataLoader
from pocket_dataset import ProteinPocketDataset


def main():
    # Set up paths
    data_dir = "crossdocked/crossdocked_pocket10"

    # Create dataset
    dataset = ProteinPocketDataset(
        root_dir=data_dir,
        split="train",
        max_ligand_atoms=128,
        max_pocket_atoms=512,
        radius=6.0,
        remove_h=True,
    )

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Get a batch
    batch = next(iter(dataloader))

    # Print batch contents
    print("\nBatch keys:", batch.keys())

    # Explore positions data (3D coordinates)
    print("\nPositions data (3D coordinates):")
    print("Shape:", batch["positions"].shape)
    print("Type:", batch["positions"].dtype)
    print("Min value:", batch["positions"].min().item())
    print("Max value:", batch["positions"].max().item())

    # Explore atom types (one-hot encoding)
    print("\nAtom types (one-hot encoding):")
    print("Shape:", batch["one_hot"].shape)
    print("Type:", batch["one_hot"].dtype)
    print("Unique values:", torch.unique(batch["one_hot"]))

    # Explore charges
    print("\nCharges:")
    print("Shape:", batch["charges"].shape)
    print("Type:", batch["charges"].dtype)
    print("Unique values:", torch.unique(batch["charges"]))

    # Explore masks
    print("\nAtom mask (ligand atoms to be diffused):")
    print("Shape:", batch["atom_mask"].shape)
    print("Type:", batch["atom_mask"].dtype)
    print("Number of True values:", batch["atom_mask"].sum().item())

    print("\nPocket mask (fixed pocket atoms):")
    print("Shape:", batch["pocket_mask"].shape)
    print("Type:", batch["pocket_mask"].dtype)
    print("Number of True values:", batch["pocket_mask"].sum().item())

    print("\nEdge mask (allowed edges for radius graph):")
    print("Shape:", batch["edge_mask"].shape)
    print("Type:", batch["edge_mask"].dtype)
    print("Number of True values:", batch["edge_mask"].sum().item())

    # Summarize ligand and pocket atom coordinates
    sample_idx = 0
    positions = batch["positions"][sample_idx]
    ligand_mask = batch["atom_mask"][sample_idx].squeeze(-1).bool()
    pocket_mask = batch["pocket_mask"][sample_idx].squeeze(-1).bool()
    ligand_positions = positions[ligand_mask]
    pocket_positions = positions[pocket_mask]
    print("\nNumber of ligand atoms:", ligand_positions.shape[0])
    print("Number of pocket atoms:", pocket_positions.shape[0])
    if ligand_positions.shape[0] > 0:
        print("Ligand coordinates range:")
        print(
            "  X: [{:.2f}, {:.2f}]".format(
                ligand_positions[:, 0].min().item(), ligand_positions[:, 0].max().item()
            )
        )
        print(
            "  Y: [{:.2f}, {:.2f}]".format(
                ligand_positions[:, 1].min().item(), ligand_positions[:, 1].max().item()
            )
        )
        print(
            "  Z: [{:.2f}, {:.2f}]".format(
                ligand_positions[:, 2].min().item(), ligand_positions[:, 2].max().item()
            )
        )
    if pocket_positions.shape[0] > 0:
        print("Pocket coordinates range:")
        print(
            "  X: [{:.2f}, {:.2f}]".format(
                pocket_positions[:, 0].min().item(), pocket_positions[:, 0].max().item()
            )
        )
        print(
            "  Y: [{:.2f}, {:.2f}]".format(
                pocket_positions[:, 1].min().item(), pocket_positions[:, 1].max().item()
            )
        )
        print(
            "  Z: [{:.2f}, {:.2f}]".format(
                pocket_positions[:, 2].min().item(), pocket_positions[:, 2].max().item()
            )
        )


if __name__ == "__main__":
    main()
