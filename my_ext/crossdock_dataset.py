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

ATOM_TYPES = ["H", "C", "N", "O", "F"]
ATOM_TYPE_TO_IDX = {sym: i for i, sym in enumerate(ATOM_TYPES)}


def one_hot(sym: str, *, n=len(ATOM_TYPES)) -> np.ndarray:
    v = np.zeros(n, dtype=np.float32)
    v[ATOM_TYPE_TO_IDX.get(sym.capitalize(), 0)] = 1.0
    return v


# -----------------------------------------------------------------------------
# Low-level file readers
# -----------------------------------------------------------------------------
def read_ligand_sdf(path: Path, *, remove_h=False):
    mol = Chem.SDMolSupplier(str(path), removeHs=remove_h, sanitize=False)[0]
    if mol is None:
        raise ValueError(f"RDKit failed for {path}")
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    pos = np.asarray(mol.GetConformer().GetPositions(), np.float32)
    x = np.stack([one_hot(a.GetSymbol()) for a in mol.GetAtoms()])
    q = np.asarray([a.GetFormalCharge() for a in mol.GetAtoms()], np.int64)
    return x, pos, q


from my_ext.ESM_pocket_encoder import ESM2PocketEncoder
import train_test

encoder = ESM2PocketEncoder()
orig_train_epoch = train_test.train_epoch
orig_test = train_test.test


def extract_pocket_context(pdb_path):
    """
    Extract a 64-dimensional context embedding for a protein pocket PDB file.
    
    Args:
        pdb_path: Path to the protein pocket PDB file
        
    Returns:
        A 64-dimensional tensor representing the protein pocket context
    """
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

        # Shuffle and split into equal thirds
        rng = np.random.RandomState(seed)
        rng.shuffle(all_items)
        n = len(all_items)
        n_split = n // 3
        if split == "train":
            self.items = all_items[:n_split]
        elif split == "val":
            self.items = all_items[n_split : 2 * n_split]
        else:
            self.items = all_items[2 * n_split : 3 * n_split]
        # If n is not divisible by 3, the last split may be slightly smaller

        print(
            f"[CrossDockedPoseDataset] Split '{split}' contains {len(self.items)} elements out of {n} total."
        )

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            _, sdf, pdb = self.items[idx]
            lx, lpos, lq = read_ligand_sdf(sdf, remove_h=self.rmH)

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
            
            # Extract the 64-dimensional context embedding for the protein pocket
            context = extract_pocket_context(str(pdb))  # encoded protein pocket - shape (64,)
            
            # Create a context_node_features tensor that will be used as context conditioning for each node
            # We'll expand this to have the same size as N (max_ligand_atoms)
            context_node_features = context.unsqueeze(0).expand(N, -1)  # Shape: (N, 64)
            
            batch = {
                "x": torch.tensor(x),
                "h": torch.tensor(x),  # duplicate for GeoLDM
                "positions": torch.tensor(pos),
                "charges": torch.tensor(q).unsqueeze(-1),
                "atom_mask": torch.tensor(lig_mask).squeeze(-1),
                "edge_index": edge_index,  # variable length
                "context": context,  # Original context vector (64,)
                "context_node_features": context_node_features,  # New field: expanded to (N, 64)
                # "pdb_path": str(pdb),  # Only keep the path for later use
                "edge_mask": torch.tensor(edge_mask),
                "one_hot": torch.tensor(x),
            }
            return batch
        except Exception as e:
            print(f"[WARN] skipping sample {self.items[idx][0]} – {e}")
            return self.__getitem__((idx + 1) % len(self))


# -----------------------------------------------------------------------------
# Collate & loaders
# -----------------------------------------------------------------------------
def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader that processes a batch of samples.
    
    This function handles the following special cases:
    1. Edge indices are concatenated with appropriate offsets
    2. Context embeddings (both global and per-node) are properly formatted:
       - Raw context vector (64-dim) is expanded to each atom in the molecule
       - Per-node context features are stacked across the batch
    3. Edge masks are constructed based on distance thresholds
    
    Args:
        samples: List of dictionaries, each representing a sample
        
    Returns:
        A dictionary with batched tensors
    """
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
        elif k == "context_node_features":
            # Stack the context_node_features tensors from each sample
            out["context_node_features"] = torch.stack([s["context_node_features"] for s in samples], 0)  # (B, N, 64)
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
    subset: Optional[int] = None,
    subset_seed=0,
    **kws,
):
    print("=== MY_EXT get_dataloaders CALLED ===")
    tr = CrossDockedPoseDataset(root_dir, split="train", **kws)
    va = CrossDockedPoseDataset(root_dir, split="val", **kws)
    te = CrossDockedPoseDataset(root_dir, split="test", **kws)
    # Remove Subset wrappers
    tr, va, te = (_maybe_subset(ds, subset, subset_seed) for ds in (tr, va, te))

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
    root = "crossdocked/crossdocked_pocket10"
    loaders = get_dataloaders(root, batch_size=2, subset=4)
    batch = next(iter(loaders["train"]))
    for k, v in batch.items():
        if k == "pdb_path":
            print(k, v)
            continue
        print(k, tuple(v.shape))

    """
    x (2, 640, 28) -> 640 atoms -> each atom one of 28 types
    h (2, 640, 28)
    edge_index (2, 2, 0) -> where are the edges
    positions (2, 640, 3) -> 640 atoms -> each atom in 3D space
    charges (2, 640, 1) -> 640 charges
    context (2, 64) -> ESM2 output 64 dimensions (global context)
    context_node_features (2, 640, 64) -> 64-dim context for each atom
    atom_mask (2, 640, 1) -> Binary Mask: where is the ligand
    pocket_mask (2, 640, 1) -> Binary Mask: where is the Pocket
    """
