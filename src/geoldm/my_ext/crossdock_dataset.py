# my_ext/crossdock_dataset.py
from __future__ import annotations
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from rdkit import Chem
# from Bio.PDB import PDBParser  # No longer needed since protein features are not used

sys.path.append(str(Path(__file__).resolve().parent.parent))

# QM9 only constits of 5 Atom types therefore we must filter
""" ATOM_TYPES = [
    "H",
    "Li",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Mg",
    "Si",
    "P",
    "S",
    "Cl",
    "Sc",
    "V",
    "Cr",
    "Fe",
    "Co",
    "Cu",
    "Se",
    "Br",
    "Y",
    "Mo",
    "Ru",
    "Sn",
    "I",
    "W",
    "Au",
    "Hg",
    "As",
] """

ATOM_TYPES = ["H", "C", "N", "O", "F"]
ATOM_TYPE_TO_IDX = {sym: i for i, sym in enumerate(ATOM_TYPES)}

# Create an inverse mapping from one-hot encoding to atom type strings
IDX_TO_ATOM_TYPE = {i: sym for i, sym in enumerate(ATOM_TYPES)}


def one_hot(sym: str, *, n=len(ATOM_TYPES)) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    if sym.capitalize() not in ATOM_TYPE_TO_IDX:
        raise ValueError(f"Atom type '{sym}' not found in ATOM_TYPES")
    v[ATOM_TYPE_TO_IDX[sym.capitalize()]] = 1.0
    return v


# -----------------------------------------------------------------------------
# Low-level file readers
# -----------------------------------------------------------------------------
def read_ligand_sdf(path: Path, *, remove_h=False):
    mol = Chem.SDMolSupplier(str(path), removeHs=remove_h, sanitize=False)[0]
    if mol is None:
        raise ValueError(f"RDKit failed for {path}")

    # Check if the molecule contains any atoms not in ATOM_TYPES
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol().capitalize()
        if sym not in ATOM_TYPES:
            raise ValueError(f"Molecule contains unsupported atom type: {sym}")

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    pos = np.asarray(mol.GetConformer().GetPositions(), np.float32)
    x = np.stack([one_hot(a.GetSymbol()) for a in mol.GetAtoms()])
    q = np.asarray([a.GetFormalCharge() for a in mol.GetAtoms()], np.int64)
    return x, pos, q


from src.geoldm.my_ext.ESM_pocket_encoder import ESM2PocketEncoder

encoder = ESM2PocketEncoder()


def extract_pocket_context(pdb_path):
    return encoder.encode_pdb(Path(pdb_path))


# --- Protein pocket reading is not needed for now; comment out for later use ---
# def read_pocket_pdb(path: Path):
#     struct = PDBParser(QUIET=True).get_structure(path.stem, str(path))
#     atoms, pos = [], []
#     for a in struct.get_atoms():
#         s = a.element.strip().capitalize()
#         if s:  # skip blanks
#             atoms.append(one_hot(s))
#             pos.append(a.coord)
#     x = np.stack(atoms).astype(np.float32)
#     pos = np.asarray(pos, np.float32)
#     q = np.zeros(len(pos), dtype=np.int64)  # no charges for protein
#     return x, pos, q


# -----------------------------------------------------------------------------
# Main dataset
# -----------------------------------------------------------------------------
class CrossDockedPoseDataset(Dataset):
    """Pocket-ligand pairs from CrossDocked2020 - pocket10 cutout.

    Output dict keys:
        x, h, positions, charges, atom_mask, pocket_mask, edge_index
    Compatible with GeoLDM latent-diffusion training.
    """

    POCKET_SUFFIX = "_pocket10.pdb"

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        max_ligand_atoms: int = 128,
        max_pocket_atoms: int = 512,
        radius: float = 6.0,
        remove_h: bool = False,
        seed: int = 0,
        folders: Optional[Sequence[str]] = None,
        poses_per_site: Optional[int] = None,
    ):
        print(f"=== MY_EXT CrossDockedPoseDataset INIT CALLED for split {split} ===")
        self.root = Path(root)
        self.max_l = max_ligand_atoms
        self.max_p = max_pocket_atoms
        self.R = radius
        self.rmH = remove_h
        self.max_retries = 200  # Limit retries to prevent infinite recursion

        # ---------- build (pose_id, sdf, pocket) index ---------------------
        site_dirs = (
            [self.root / f for f in folders]
            if folders
            else [p for p in self.root.iterdir() if p.is_dir()]
        )
        site_dirs = sorted(site_dirs)
        all_items: List[Tuple[str, Path, Path]] = []
        for site in site_dirs:
            sdf_files = sorted(site.glob("*.sdf"))
            if poses_per_site:
                sdf_files = sdf_files[:poses_per_site]
            for sdf in sdf_files:
                pocket = site / f"{sdf.stem}{self.POCKET_SUFFIX}"
                if pocket.exists():
                    all_items.append((f"{site.name}/{sdf.stem}", sdf, pocket))

        # Shuffle and split into thirds by default, or using standard 90/5/5 split
        rng = np.random.RandomState(seed)
        rng.shuffle(all_items)
        n = len(all_items)

        # Allow for different split ratios
        split_mode = getattr(self, "split_mode", "equal")  # Default to equal splits
        split_mode = "equal"
        if split_mode == "equal":
            # Equal thirds split
            n_split = n // 3
            if split == "train":
                self.items = all_items[:n_split]
            elif split == "val":
                self.items = all_items[n_split : 2 * n_split]
            else:  # test
                self.items = all_items[2 * n_split : 3 * n_split]
        else:
            # Standard 90/5/5 split
            train_end = int(0.9 * n)
            val_end = train_end + int(0.05 * n)

            if split == "train":
                self.items = all_items[:train_end]
            elif split == "val":
                self.items = all_items[train_end:val_end]
            else:  # test
                self.items = all_items[val_end:]

        print(
            f"[CrossDockedPoseDataset] Split '{split}' contains {len(self.items)} elements out of {n} total."
        )

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int, retry_count: int = 0) -> Dict[str, torch.Tensor]:
        if retry_count >= self.max_retries:
            raise RuntimeError(
                f"Maximum retries ({self.max_retries}) exceeded. Could not find a valid molecule."
            )

        try:
            _, sdf, pdb = self.items[idx]
            try:
                lx, lpos, lq = read_ligand_sdf(sdf, remove_h=self.rmH)
            except ValueError as e:
                if "unsupported atom type" in str(e):
                    # Skip molecules with unsupported atom types and try the next one
                    # print(f"[FILTER] Skipping {sdf.name}: {e}")
                    return self.__getitem__((idx + 1) % len(self), retry_count + 1)
                else:
                    raise  # Re-raise other ValueError types
            if retry_count > 10:
                print(f"retries {retry_count}")
            # truncate to budgets
            lx, lpos, lq = lx[: self.max_l], lpos[: self.max_l], lq[: self.max_l]

            n_l = len(lpos)
            N = self.max_l  # Only ligand atoms
            x = np.zeros((N, len(ATOM_TYPES)), np.float32)
            pos = np.zeros((N, 3), np.float32)
            q = np.zeros((N,), np.int64)
            lig_mask = np.zeros((N, 1), np.float32)

            # fill ligand block
            x[:n_l] = lx
            pos[:n_l] = lpos
            q[:n_l] = lq
            lig_mask[:n_l] = 1.0

            # radius graph on REAL atoms only
            N_real = n_l
            dist = np.linalg.norm(lpos[:, None, :] - lpos[None, :, :], axis=-1)
            mask = (dist < 4.5) & (dist > 0)  # exclude self-edges
            edge_mask = np.zeros((self.max_l, self.max_l), dtype=bool)
            edge_mask[:N_real, :N_real] = mask

            edge_index = torch.zeros((2, 0), dtype=torch.long)  # Dummy edge_index
            context = extract_pocket_context(str(pdb))  # encoded protein pocket
            batch = {
                "x": torch.tensor(x),
                "h": torch.tensor(x),  # duplicate for GeoLDM
                "positions": torch.tensor(pos),
                "charges": torch.tensor(q).unsqueeze(-1),
                "atom_mask": torch.tensor(lig_mask).squeeze(-1),
                "edge_index": edge_index,  # variable length
                "context": context,
                # "pdb_path": str(pdb),  # Only keep the path for later use
                "edge_mask": torch.tensor(edge_mask),
                "one_hot": torch.tensor(x),
            }
            return batch
        except Exception as e:
            print(f"[WARN] skipping sample {self.items[idx][0]} – {e}")
            return self.__getitem__((idx + 1) % len(self), retry_count + 1)


# -----------------------------------------------------------------------------
# Collate & loaders
# -----------------------------------------------------------------------------
def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    print("=== MY_EXT COLLATE_FN CALLED ===")
    out: Dict[str, torch.Tensor] = {}
    for k in samples[0]:
        if k == "edge_index":
            continue
        if k == "context":
            context = samples[0]["context"]
            if context.dim() == 1:
                # (D,) -> (N, D)
                N = samples[0]["x"].shape[0]
                context_expanded = [s["context"].unsqueeze(0).expand(N, -1) for s in samples]
                out["context"] = torch.stack(context_expanded, 0)  # (B, N, D)
            elif context.dim() == 2:
                # (N, D)
                out["context"] = torch.stack([s["context"] for s in samples], 0)
            else:
                raise ValueError("context must be 1D or 2D tensor per sample")
        else:
            out[k] = torch.stack([s[k] for s in samples], 0)

    N = samples[0]["x"].shape[0]
    edges = [s["edge_index"] + i * N for i, s in enumerate(samples)]
    out["edge_index"] = torch.cat(edges, dim=1)  # (2, ΣE)

    # Construct a valid edge_mask based on distance threshold
    positions = out["positions"]  # (B, N, 3)
    B = positions.shape[0]
    edge_masks = []
    threshold = 4.5  # Angstroms
    for b in range(B):
        pos = positions[b]  # (N, 3)
        dist = torch.cdist(pos, pos)  # (N, N)
        mask = (dist < threshold) & (dist > 0)  # exclude self-edges
        edge_masks.append(mask)
    out["edge_mask"] = torch.stack(edge_masks, 0)  # (B, N, N)

    # Squeeze the last dimension to get [B, N] shape
    out["atom_mask"] = out["atom_mask"]
    out["node_mask"] = out["atom_mask"]  # shape (B, N)
    out["pocket_mask"] = out["pocket_mask"]

    print(f"[DEBUG] x shape: {out['x'].shape}")
    print(f"[DEBUG] atom_mask shape: {out['atom_mask'].shape}, ndim: {out['atom_mask'].ndim}")
    print(f"[DEBUG] node_mask shape: {out['node_mask'].shape}, ndim: {out['node_mask'].ndim}")
    return out


def _maybe_subset(ds: Dataset, n: Optional[int], seed=0):
    if n is None or n >= len(ds):
        return ds
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    return Subset(ds, idx.tolist())


def get_dataloaders(
    root_dir: str,
    *,
    batch_size=8,
    num_workers=0,
    **kws,
):
    print("=== MY_EXT get_dataloaders CALLED ===")
    tr = CrossDockedPoseDataset(root_dir, split="train", **kws)
    va = CrossDockedPoseDataset(root_dir, split="val", **kws)
    te = CrossDockedPoseDataset(root_dir, split="test", **kws)
    # Remove Subset wrappers

    def make_loader(ds, shuf):
        return DataLoader(
            ds, batch_size, shuffle=shuf, collate_fn=collate_fn, num_workers=num_workers
        )

    return {
        "train": make_loader(tr, True),
        "valid": make_loader(va, False),  # Must be 'valid' for main_qm9.py compatibility
        "test": make_loader(te, False),
    }


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    root = "crossdocked/crossdocked_pocket_debug"
    loaders = get_dataloaders(root, batch_size=1)
    batch = next(iter(loaders["train"]))
    for k, v in batch.items():
        if k == "pdb_path":
            print(k, v)
            continue
        print(k, tuple(v.shape))

    """
    x (2, 640, 29) -> 640 atoms -> each atom one of 29 types
    h (2, 640, 29)
    edge_index (2, 2, 0) -> where are the edges
    positions (2, 640, 3) -> 640 atoms -> each atom in 3D space
    charges (2, 640, 1) -> 640 charges
    context (2, 64) -> ESM2 output 64 dimensions
    atom_mask (2, 640, 1) -> Binary Mask: where is the ligand
    pocket_mask (2, 640, 1) -> Binary Mask: where is the Pocket
    """
