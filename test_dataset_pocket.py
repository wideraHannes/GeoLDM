from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict
from torch_geometric.loader import DataLoader

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from rdkit import Chem
from Bio.PDB import PDBParser

# ATOM TYPES supported in this dataset
ATOM_TYPES = [
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "Na",
    "K",
    "Ca",
    "Mg",
    "Zn",
    "Fe",
    "B",
    "Se",
    "Si",
    "Mn",
    "Co",
    "Cu",
    "Ni",
    "V",
    "Cr",
    "Hg",
    "Pb",
    "Al",
]
ATOM_TYPE_TO_IDX = {a: i for i, a in enumerate(ATOM_TYPES)}


def one_hot(element: str, num_classes: int = len(ATOM_TYPES)) -> np.ndarray:
    """Convert element string to one-hot encoding"""
    idx = ATOM_TYPE_TO_IDX.get(element.capitalize(), 0)  # Default to first element if not found
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[idx] = 1.0
    return vec


# ---------------------------- low-level readers ----------------------------- #


def read_ligand_sdf(
    sdf_path: Path, remove_h: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ligand SDF file and return atom features, positions, and charges"""
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=remove_h, sanitize=False)[0]
    if mol is None:
        raise ValueError(f"Failed to read molecule from {sdf_path}")

    # Kekulize to convert aromatic bonds to single/double
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass  # Skip Kekulization if it fails

    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions(), dtype=np.float32)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    x = np.stack([one_hot(s) for s in symbols])
    charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=np.int64)

    return x, pos, charges


def read_pocket_pdb(pdb_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read pocket PDB file and return atom features, positions, and charges"""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))

    atoms, coords, charge = [], [], []
    for atom in struct.get_atoms():
        sym = atom.element.strip().capitalize()
        if sym:  # ignore weird/blank records
            atoms.append(sym)
            coords.append(atom.coord)
            charge.append(0)  # Default charge for pocket atoms is 0

    x = np.stack([one_hot(e) for e in atoms])
    pos = np.array(coords, dtype=np.float32)
    charges = np.array(charge, dtype=np.int64)

    return x, pos, charges


# ----------------------------- the main Dataset ---------------------------- #
class CrossDockedPoseDataset(Dataset):
    """Dataset for CrossDocked protein-ligand complexes, compatible with GeoLDM training"""

    POCKET_SUFFIX = "_pocket10.pdb"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        max_ligand_atoms: int = 128,
        max_pocket_atoms: int = 512,
        radius: float = 6.0,
        remove_h: bool = False,
        seed: int = 0,
        folders: Optional[Sequence[str]] = None,
        poses_per_site: Optional[int] = None,
        device: torch.device | str | None = None,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Path to the *crossdocked_pocket10* directory.
        split : str
            Dataset split: "train", "val", or "test"
        max_ligand_atoms : int
            Maximum number of ligand atoms to include
        max_pocket_atoms : int
            Maximum number of pocket atoms to include
        radius : float
            Radius for edge connections in Angstroms
        remove_h : bool
            Whether to remove hydrogen atoms
        seed : int
            Random seed for splitting data
        folders : list[str], optional
            Restrict to these binding-site folders. By default every folder is used.
        poses_per_site : int, optional
            Sub-sample at most N poses per site (deterministic, first N).
        """
        self.root = Path(root)
        self.max_lig = max_ligand_atoms
        self.max_poc = max_pocket_atoms
        self.radius = radius
        self.remove_h = remove_h
        self.device = torch.device(device) if device else torch.device("cpu")

        # --------- build an index of every (ligand.sdf, pocket.pdb) pair ------ #
        site_dirs = (
            [self.root / f for f in folders]
            if folders
            else sorted([p for p in self.root.iterdir() if p.is_dir()])
        )

        rng = np.random.RandomState(seed)
        all_sites = sorted(site_dirs)
        rng.shuffle(all_sites)
        n = len(all_sites)

        # Split the dataset
        if split == "train":
            site_dirs = all_sites[: int(0.8 * n)]
        elif split == "val":
            site_dirs = all_sites[int(0.8 * n) : int(0.9 * n)]
        else:  # test
            site_dirs = all_sites[int(0.9 * n) :]

        self.items: list[tuple[str, Path, Path]] = []  # (pose_id, sdf, pdb)
        for site in site_dirs:
            # all .sdf files = all **poses** for that site
            sdf_files = sorted(site.glob("*.sdf"))
            if poses_per_site:
                sdf_files = sdf_files[:poses_per_site]

            for sdf in sdf_files:
                # matching pocket pdb must have same stem + _pocket10.pdb
                stem = sdf.stem  # e.g. 1j0c_..._docked_3
                pocket = site / f"{stem}{self.POCKET_SUFFIX}"
                if not pocket.exists():
                    continue  # Skip if pocket doesn't exist
                pose_id = f"{site.name}/{stem}"
                self.items.append((pose_id, sdf, pocket))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            pose_id, sdf_path, pdb_path = self.items[idx]

            # Read ligand and pocket
            lig_x, lig_pos, lig_q = read_ligand_sdf(sdf_path, self.remove_h)
            poc_x, poc_pos, poc_q = read_pocket_pdb(pdb_path)

            # Truncate if needed to match max sizes
            lig_x = lig_x[: self.max_lig]
            lig_pos = lig_pos[: self.max_lig]
            lig_q = lig_q[: self.max_lig]

            poc_x = poc_x[: self.max_poc]
            poc_pos = poc_pos[: self.max_poc]
            poc_q = poc_q[: self.max_poc]

            # Get actual sizes
            N_lig = lig_pos.shape[0]
            N_poc = poc_pos.shape[0]
            N = self.max_lig + self.max_poc

            # Create empty arrays
            one_hot_array = np.zeros((N, len(ATOM_TYPES)), dtype=np.float32)
            positions = np.zeros((N, 3), dtype=np.float32)
            charges = np.zeros((N,), dtype=np.int64)  # Charges as 1D array
            atom_mask = np.zeros((N, 1), dtype=np.float32)
            pocket_mask = np.zeros((N, 1), dtype=np.float32)

            # Fill arrays
            # Place ligand atoms at the beginning
            positions[:N_lig] = lig_pos
            one_hot_array[:N_lig] = lig_x
            charges[:N_lig] = lig_q
            atom_mask[:N_lig] = 1.0  # ligand atoms will be diffused

            # Place pocket atoms after the max ligand atoms
            positions[self.max_lig : self.max_lig + N_poc] = poc_pos
            one_hot_array[self.max_lig : self.max_lig + N_poc] = poc_x
            charges[self.max_lig : self.max_lig + N_poc] = poc_q
            pocket_mask[self.max_lig : self.max_lig + N_poc] = 1.0

            # Build edge mask (radius graph)
            diff = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            edge_mask = (dists < self.radius) & (dists > 0.01)

            # Convert to tensors
            sample = {
                "positions": torch.from_numpy(positions),
                "one_hot": torch.from_numpy(one_hot_array),
                "charges": torch.from_numpy(charges).unsqueeze(-1),  # Make it (N, 1)
                "atom_mask": torch.from_numpy(atom_mask),
                "pocket_mask": torch.from_numpy(pocket_mask),
                "edge_mask": torch.from_numpy(edge_mask.astype(np.float32)),
            }

            return {k: v.to(self.device) for k, v in sample.items()}

        except Exception as e:
            print(f"[WARN] Skipping {self.items[idx][0]} due to error: {e}")
            # Return next sample (cycle if at end)
            return self.__getitem__((idx + 1) % len(self.items))


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    keys = samples[0].keys()
    batch = {k: torch.stack([s[k] for s in samples], dim=0) for k in keys}
    return batch


# -----------------------------  helper ------------------------------------- #


def _maybe_subset(dataset: Dataset, subset_size: Optional[int] = None, seed: int = 0) -> Dataset:
    """Return a Subset of the dataset if *subset_size* is specified and smaller than len(dataset)."""
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices.tolist())


# --------------------------- public API ------------------------------------ #


def get_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    *,
    subset: Optional[int] = None,
    subset_seed: int = 0,
    **kwargs,
):
    """Create train, validation, and test DataLoaders.

    Parameters
    ----------
    root_dir : str
        Path to the *crossdocked_pocket10* directory.
    batch_size : int
        Batch size for the DataLoaders.
    num_workers : int
        Number of worker processes for the DataLoaders.
    subset : int, optional
        If given, each split will be randomly downsampled to *subset* examples.
        Useful for fast debugging & development.
    subset_seed : int
        Random seed controlling which samples are selected when *subset* is used.
    **kwargs : dict
        Extra arguments forwarded to :class:`CrossDockedPoseDataset`.
    """

    train_set = CrossDockedPoseDataset(root_dir, split="train", **kwargs)
    val_set = CrossDockedPoseDataset(root_dir, split="val", **kwargs)
    test_set = CrossDockedPoseDataset(root_dir, split="test", **kwargs)

    # Optionally shrink datasets for debugging
    train_set = _maybe_subset(train_set, subset, seed=subset_seed)
    val_set = _maybe_subset(val_set, subset, seed=subset_seed)
    test_set = _maybe_subset(test_set, subset, seed=subset_seed)

    loaders = {
        "train": DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            val_set,
            batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            test_set,
            batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
        ),
    }
    return loaders


if __name__ == "__main__":
    # Example usage: load a single batch and print keys/shapes
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "./crossdocked/crossdocked_pocket10"
    loaders = get_dataloaders(
        root,
        batch_size=2,
        max_ligand_atoms=128,
        max_pocket_atoms=512,
        subset=1,  # Debug with just 10 samples per split
    )

    batch = next(iter(loaders["train"]))
    print("Loaded batch from:", root)
    for k, v in batch.items():
        print(f"{k}: shape {tuple(v.shape)} | dtype {v.dtype}")
