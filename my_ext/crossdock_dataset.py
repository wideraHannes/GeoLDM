# my_ext/crossdock_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from rdkit import Chem
from Bio.PDB import PDBParser

# -----------------------------------------------------------------------------
# Atom vocabulary
# -----------------------------------------------------------------------------
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


def read_pocket_pdb(path: Path):
    struct = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    atoms, pos = [], []
    for a in struct.get_atoms():
        s = a.element.strip().capitalize()
        if s:  # skip blanks
            atoms.append(one_hot(s))
            pos.append(a.coord)
    x = np.stack(atoms).astype(np.float32)
    pos = np.asarray(pos, np.float32)
    q = np.zeros(len(pos), dtype=np.int64)  # no charges for protein
    return x, pos, q


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
        rng = np.random.RandomState(seed)
        rng.shuffle(site_dirs)
        n = len(site_dirs)
        if split == "train":
            site_dirs = site_dirs[: int(0.8 * n)]
        elif split == "val":
            site_dirs = site_dirs[int(0.8 * n) : int(0.9 * n)]
        else:
            site_dirs = site_dirs[int(0.9 * n) :]

        self.items: List[Tuple[str, Path, Path]] = []
        for site in site_dirs:
            sdf_files = sorted(site.glob("*.sdf"))
            if poses_per_site:
                sdf_files = sdf_files[:poses_per_site]
            for sdf in sdf_files:
                pocket = site / f"{sdf.stem}{self.POCKET_SUFFIX}"
                if pocket.exists():
                    self.items.append((f"{site.name}/{sdf.stem}", sdf, pocket))

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            _, sdf, pdb = self.items[idx]
            lx, lpos, lq = read_ligand_sdf(sdf, remove_h=self.rmH)
            px, ppos, pq = read_pocket_pdb(pdb)

            # truncate to budgets
            lx, lpos, lq = lx[: self.max_l], lpos[: self.max_l], lq[: self.max_l]
            px, ppos, pq = px[: self.max_p], ppos[: self.max_p], pq[: self.max_p]

            n_l, n_p = len(lpos), len(ppos)
            N = self.max_l + self.max_p  # 640
            x = np.zeros((N, len(ATOM_TYPES)), np.float32)
            pos = np.zeros((N, 3), np.float32)
            q = np.zeros((N,), np.int64)
            lig_mask = np.zeros((N, 1), np.float32)
            pocket_mask = np.zeros((N, 1), np.float32)

            # fill ligand block
            x[:n_l] = lx
            pos[:n_l] = lpos
            q[:n_l] = lq
            lig_mask[:n_l] = 1.0

            # fill pocket block (starts at self.max_l to keep indices stable)
            start = self.max_l
            x[start : start + n_p] = px
            pos[start : start + n_p] = ppos
            q[start : start + n_p] = pq
            pocket_mask[start : start + n_p] = 1.0

            # radius graph on REAL atoms only

            # real_idx = np.arange(n_l + n_p)
            edge_index = torch.zeros((2, 0), dtype=torch.long)  # Dummy edge_index
            return {
                "x": torch.tensor(x),
                "h": torch.tensor(x),  # duplicate for GeoLDM
                "positions": torch.tensor(pos),
                "charges": torch.tensor(q).unsqueeze(-1),
                "atom_mask": torch.tensor(lig_mask),
                "pocket_mask": torch.tensor(pocket_mask),
                "edge_index": edge_index,  # variable length
            }
        except Exception as e:
            print(f"[WARN] skipping sample {self.items[idx][0]} – {e}")
            return self.__getitem__((idx + 1) % len(self))


# -----------------------------------------------------------------------------
# Collate & loaders
# -----------------------------------------------------------------------------
def collate_fn(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stacks fixed-size tensors, concatenates edge lists with node offsets."""
    out: Dict[str, torch.Tensor] = {}
    for k in samples[0]:
        if k == "edge_index":
            continue
        out[k] = torch.stack([s[k] for s in samples], 0)

    N = samples[0]["x"].shape[0]
    edges = [s["edge_index"] + i * N for i, s in enumerate(samples)]
    out["edge_index"] = torch.cat(edges, dim=1)  # (2, ΣE)
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
    tr = CrossDockedPoseDataset(root_dir, split="train", **kws)
    va = CrossDockedPoseDataset(root_dir, split="val", **kws)
    te = CrossDockedPoseDataset(root_dir, split="test", **kws)
    tr, va, te = (_maybe_subset(ds, subset, subset_seed) for ds in (tr, va, te))
    make = lambda ds, shuf: DataLoader(
        ds, batch_size, shuffle=shuf, collate_fn=collate_fn, num_workers=num_workers
    )
    return {"train": make(tr, True), "val": make(va, False), "test": make(te, False)}


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    root = "crossdocked/crossdocked_pocket10"
    loaders = get_dataloaders(root, batch_size=2, subset=4)
    batch = next(iter(loaders["train"]))
    for k, v in batch.items():
        print(k, tuple(v.shape))

    """
    x (2, 640, 28) -> 640 atoms -> each atom one of 28 types
    h (2, 640, 28)
    positions (2, 640, 3) -> 640 atoms -> each atom in 3D space
    charges (2, 640, 1) -> 640 charges
    atom_mask (2, 640, 1) -> Binary Mask: where is the ligand
    pocket_mask (2, 640, 1) -> Binary Mask: where is the Pocket
    edge_index (2, 2, 0) -> where are the edges
    """
