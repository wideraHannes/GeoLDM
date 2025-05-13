"""
ProteinPocketDataset
====================
Utility classes and helper functions to load protein–ligand complexes from
CrossDocked2020 (or any PDB‐bind‑style directory) and prepare tensors that are
compatible with GeoLDM.  The key idea is to treat pocket atoms as **fixed
context nodes**: they are concatenated to the ligand atoms in every sample, but
flagged by a `pocket_mask` so that the diffusion model never perturbs their
coordinates or atom‑type logits – they only act as conditioning geometry.

The dataset yields a dict with the following keys:
    positions        – (N, 3) float32 tensor of Å coordinates.
    one_hot          – (N, A) int/float tensor of atom‑type one‑hot (A = 28 by default).
    charges          – (N, 1) integer tensor of formal charges (0 for most proteins).
    atom_mask        – (N, 1) bool tensor; 1 = ligand atom to be diffused.
    pocket_mask      – (N, 1) bool tensor; 1 = pocket atom (kept *fixed*).
    edge_mask        – (N, N) bool tensor; 1 = edge allowed (used for radius graph).

During sampling, you pass **node_mask = atom_mask** (ligand only) to GeoLDM so
its noise schedule and score network apply solely to ligand atoms.  The pocket
atoms stay frozen but fully interact via EGNN message passing.

Author: ChatGPT (OpenAI o3) – May 2025
"""

from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser
from rdkit import Chem
import warnings

# -----------------------------
# Basic chemistry utilities
# -----------------------------

ATOM_TYPES = [
    "H",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
    "Se",
    "Si",
    "Zn",
    "Fe",
    "Mg",
    "Ca",
    "Mn",
    "K",
    "Na",
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


def one_hot(atom_symbol: str, num_classes: int = len(ATOM_TYPES)) -> np.ndarray:
    idx = ATOM_TYPE_TO_IDX.get(atom_symbol.capitalize(), None)
    vec = np.zeros(num_classes, dtype=np.float32)
    if idx is not None:
        vec[idx] = 1.0
    return vec


# -----------------------------
# Pocket extraction helpers
# -----------------------------


def extract_pocket_atoms(structure, ligand_resname: str, radius: float = 6.0, lig_coords=None):
    """Return atom list (Bio.PDB atoms) within *radius* Å of any ligand atom.
    If ligand_resname is not found, try spatial matching to lig_coords
    (if provided).
    """
    ligand_atoms = [a for a in structure.get_atoms() if a.parent.get_resname() == ligand_resname]
    if not ligand_atoms and lig_coords is not None:
        # Fallback: spatial match (find atoms within 1.5A of any ligand atom)
        from Bio.PDB import NeighborSearch

        all_atoms = list(structure.get_atoms())
        ns = NeighborSearch(all_atoms)
        matched = set()
        for coord in lig_coords:
            close = ns.search(coord, 1.5, level="A")
            matched.update(close)
        ligand_atoms = list(matched)
        if ligand_atoms:
            warnings.warn(
                f"Ligand residue '{ligand_resname}' not found, "
                "using spatial match for ligand atoms."
            )
    if not ligand_atoms:
        raise ValueError(
            f"Ligand {ligand_resname} not found in structure and no spatial match possible."
        )
    # KD‑tree search using Bio.PDB.NeighborSearch
    from Bio.PDB import NeighborSearch

    ns = NeighborSearch(list(structure.get_atoms()))
    pocket_atoms = set()
    for latom in ligand_atoms:
        pocket_atoms.update(ns.search(latom.coord, radius, level="A"))
    # Remove ligand atoms themselves
    pocket_atoms = [a for a in pocket_atoms if a not in ligand_atoms]
    return ligand_atoms, pocket_atoms


# -----------------------------
# Dataset class
# -----------------------------


class ProteinPocketDataset(Dataset):
    """Minimal PyTorch Dataset wrapping CrossDocked2020 directory structure.

    Expected folder layout (simplified):
        root_dir/
            <complex_id>/
                pocket.pdb    – receptor pocket only (pre‑cropped)  [optional]
                decoy.pdb     – pre‑aligned decoy receptor          [optional]
                ligand.sdf    – bound ligand                        [required]
                complex.pdb   – full complex                        [fallback]

    The dataset auto‑parses PDB/SDF, extracts pocket atoms (if pocket.pdb not
    given), builds radius graph edge_mask, and returns padded tensors.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        max_ligand_atoms: int = 128,
        max_pocket_atoms: int = 512,
        radius: float = 6.0,
        remove_h: bool = True,
        seed: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_lig = max_ligand_atoms
        self.max_poc = max_pocket_atoms
        self.radius = radius
        self.remove_h = remove_h

        rng = np.random.RandomState(seed)
        all_complexes = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        rng.shuffle(all_complexes)
        n = len(all_complexes)
        if split == "train":
            self.complexes = all_complexes[: int(0.8 * n)]
        elif split == "val":
            self.complexes = all_complexes[int(0.8 * n) : int(0.9 * n)]
        else:
            self.complexes = all_complexes[int(0.9 * n) :]

    # ---------------------------------------------------------
    # Core loader – returns padded tensors ready for GeoLDM
    # ---------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cmp_dir = self.complexes[idx]
        ligand_path = next(cmp_dir.glob("*.sdf"))
        pocket_path = next(cmp_dir.glob("*_pocket10.pdb"))
        complex_path = pocket_path  # Use pocket PDB as the complex file

        try:
            # 1) Parse ligand with RDKit
            mol = Chem.SDMolSupplier(str(ligand_path), removeHs=self.remove_h)[0]
            if mol is None:
                raise RuntimeError(f"Failed to read ligand {ligand_path}")
            Chem.Kekulize(mol, clearAromaticFlags=True)
            ligand_conf = mol.GetConformer()

            lig_coords = np.array(ligand_conf.GetPositions(), dtype=np.float32)
            lig_symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            lig_q = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], dtype=np.int64)
            lig_onehot = np.stack([one_hot(s) for s in lig_symbols])

            # 2) Parse complex with Bio.PDB to get pocket atoms
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("cmp", str(complex_path))

            # Try to get ligand resname from SDF file name
            ligand_resname = None
            if mol.HasProp("_Name"):
                ligand_resname = mol.GetProp("_Name")[:3].upper()
            else:
                # Try to extract from filename
                sdf_name = ligand_path.stem
                if "_lig_" in sdf_name:
                    ligand_resname = sdf_name.split("_lig_")[1].split("_")[0].upper()

            if ligand_resname is None:
                ligand_resname = "LIG"  # Default fallback

            # Debug: print ligand_resname and all residue names in the structure
            all_resnames = set(res.get_resname() for res in structure.get_residues())
            print(f"ligand_resname: {ligand_resname}")
            print(f"All residue names in structure: {all_resnames}")

            ligand_atoms, pocket_atoms = extract_pocket_atoms(
                structure, ligand_resname, self.radius, lig_coords=lig_coords
            )

            poc_coords = np.array([a.coord for a in pocket_atoms], dtype=np.float32)
            poc_symbols = [a.element.strip() for a in pocket_atoms]
            poc_q = np.zeros(len(pocket_atoms), dtype=np.int64)
            poc_onehot = np.stack([one_hot(s) for s in poc_symbols])

            # 3) Concatenate and pad
            N_lig, N_poc = lig_coords.shape[0], poc_coords.shape[0]
            N = self.max_lig + self.max_poc
            one_hot_array = np.zeros((N, len(ATOM_TYPES)), dtype=np.float32)
            positions = np.zeros((N, 3), dtype=np.float32)
            charges = np.zeros((N, 1), dtype=np.int64)
            atom_mask = np.zeros((N, 1), dtype=np.float32)
            pocket_mask = np.zeros((N, 1), dtype=np.float32)

            positions[:N_lig] = lig_coords
            one_hot_array[:N_lig] = lig_onehot
            charges[:N_lig, 0] = lig_q
            atom_mask[:N_lig] = 1.0  # ligand atoms will be diffused

            positions[self.max_lig : self.max_lig + N_poc] = poc_coords
            one_hot_array[self.max_lig : self.max_lig + N_poc] = poc_onehot
            charges[self.max_lig : self.max_lig + N_poc, 0] = poc_q
            pocket_mask[self.max_lig : self.max_lig + N_poc] = 1.0

            # 4) Build edge mask (radius graph on all nodes)
            diff = positions[:, None, :] - positions[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            edge_mask = (dists < self.radius) & (dists > 0.01)

            sample = {
                "positions": torch.from_numpy(positions),
                "one_hot": torch.from_numpy(one_hot_array),
                "charges": torch.from_numpy(charges),
                "atom_mask": torch.from_numpy(atom_mask),
                "pocket_mask": torch.from_numpy(pocket_mask),
                "edge_mask": torch.from_numpy(edge_mask.astype(np.float32)),
            }
            return sample
        except Exception as e:
            print(f"[WARN] Skipping {cmp_dir.name} due to error: {e}")
            # Try next sample (cycle if at end)
            return self.__getitem__((idx + 1) % len(self.complexes))

    def __len__(self):
        return len(self.complexes)


# ---------------------------------------------------------
# Collate fn – merge batch and keep same API as QM9 loader
# ---------------------------------------------------------


def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = samples[0].keys()
    batch = {}
    for k in keys:
        batch[k] = torch.stack([s[k] for s in samples], dim=0)
    return batch


# Convenience dataloader wrapper


def get_dataloaders(root_dir: str, batch_size: int = 8, num_workers: int = 0):
    train_set = ProteinPocketDataset(root_dir, "train")
    val_set = ProteinPocketDataset(root_dir, "val")
    test_set = ProteinPocketDataset(root_dir, "test")

    from torch.utils.data import DataLoader

    return {
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


def main():
    # Example usage: load a single batch and print keys/shapes
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "crossdocked/crossdocked_pocket10"
    loaders = get_dataloaders(root, batch_size=2, num_workers=0)
    batch = next(iter(loaders["train"]))
    print("Loaded batch from:", root)
    for k, v in batch.items():
        print(f"{k}: shape {tuple(v.shape)} | dtype {v.dtype}")


if __name__ == "__main__":
    main()
