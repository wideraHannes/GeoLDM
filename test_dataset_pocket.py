# refined_crossdocked_dataset.py
from pathlib import Path
from typing import List, Optional, Tuple, Sequence
from torch_geometric.loader import DataLoader

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
from Bio.PDB import PDBParser

ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "Na", "K", "Ca", "Mg", "Zn", "Fe"]


def one_hot(element: str) -> List[int]:
    return [int(element == e) for e in ELEMENTS]


# ---------------------------- low-level readers ----------------------------- #
def read_ligand_sdf(sdf_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    mol = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)[0]
    conf = mol.GetConformer()
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    x = torch.tensor([one_hot(a.GetSymbol()) for a in mol.GetAtoms()], dtype=torch.float32)
    return x, pos


def read_pocket_pdb(pdb_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_path.stem, str(pdb_path))
    atoms, coords = [], []
    for atom in struct.get_atoms():
        sym = atom.element.strip().capitalize()
        if sym:  # ignore weird/blank records
            atoms.append(sym)
            coords.append(atom.coord)
    x = torch.tensor([one_hot(e) for e in atoms], dtype=torch.float32)
    pos = torch.tensor(coords, dtype=torch.float32)
    return x, pos


# ----------------------------- the main Dataset ---------------------------- #
class CrossDockedPoseDataset(Dataset):
    r"""
    Each __getitem__ returns a pose-level Data object with
        · data.x        (node features, ligand first – pocket second)
        · data.pos      (3-D coordinates)
        · data.lig_mask (bool, True for ligand atoms)
        · data.id       (unique pose id: <site>/<slug>)
        · optional **extras
    """

    POCKET_SUFFIX = "_pocket10.pdb"

    def __init__(
        self,
        root: str | Path,
        folders: Optional[Sequence[str]] = None,
        poses_per_site: Optional[int] = None,
        transform=None,
        device: torch.device | str | None = None,
    ):
        """
        Parameters
        ----------
        root : str | Path
            Path to the *crossdocked_pocket10* directory.
        folders : list[str], optional
            Restrict to these binding-site folders.  By default every folder is used.
        poses_per_site : int, optional
            Sub-sample at most N poses per site (deterministic, first N).
        """
        self.root = Path(root)
        self.transform = transform
        self.device = torch.device(device) if device else torch.device("cpu")

        # --------- build an index of every (ligand.sdf , pocket.pdb) pair ------ #
        site_dirs = (
            [self.root / f for f in folders]
            if folders
            else sorted([p for p in self.root.iterdir() if p.is_dir()])
        )

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
                    # If the user kept *_pocket10.pdb* without the ligand part,
                    # fall back to the single file in the folder.
                    raise FileNotFoundError(f"No pocket PDB for {sdf}")
                pose_id = f"{site.name}/{stem}"
                self.items.append((pose_id, sdf, pocket))

    # ------------------------------------------------------------------------ #
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        pose_id, sdf_path, pdb_path = self.items[idx]

        lig_x, lig_pos = read_ligand_sdf(sdf_path)
        poc_x, poc_pos = read_pocket_pdb(pdb_path)

        x = torch.cat([lig_x, poc_x], dim=0)
        pos = torch.cat([lig_pos, poc_pos], dim=0)

        data = Data(x=x, pos=pos)
        data.lig_mask = torch.zeros(x.size(0), dtype=torch.bool)
        data.lig_mask[: len(lig_x)] = True
        data.id = pose_id

        if self.transform:
            data = self.transform(data)
        return data.to(self.device)


if __name__ == "__main__":
    ds = CrossDockedPoseDataset(
        root="./crossdocked/crossdocked_pocket10",
        poses_per_site=5,  # keep 5 docked poses per pocket
        device="cpu",
    )
    print(f"{len(ds):,} pose samples")

    loader = DataLoader(ds, batch_size=16, shuffle=True, follow_batch=["x"])
    batch = next(iter(loader))
    print(batch)  # torch_geometric Batch
    print(batch.lig_mask)  # ligand-atom indicator
