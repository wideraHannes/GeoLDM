import torch
from pathlib import Path
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
import esm


# --- Utility: Convert pocket PDB to FASTA sequence ---
def pocket_fasta(pocket_pdb: Path) -> str:
    """Return single-letter FASTA sequence of all residues
    that have ≥1 atom in the pocket PDB file."""
    struct = PDBParser(QUIET=True).get_structure("p", str(pocket_pdb))
    res_letters = []
    for res in struct.get_residues():
        if not is_aa(res, standard=True):
            continue  # skip waters / ligands
        res_letters.append(seq1(res.get_resname()))
    return "".join(res_letters)  # e.g. "AVRLI..."


# --- ESM-2 Pocket Encoder ---
class ESM2PocketEncoder:
    """
    Encoder that uses the ESM2 protein language model to generate context embeddings for protein pockets.

    This encoder:
    1. Takes a protein pocket PDB file
    2. Extracts the amino acid sequence
    3. Passes it through ESM2 model
    4. Projects the output to a fixed 64-dimensional vector

    This vector can be used as context conditioning for ligand generation.
    """

    def __init__(self, device=None, proj_dim=64, freeze_proj=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.proj = torch.nn.Linear(640, proj_dim, bias=False).to(self.device)
        if freeze_proj:
            for p in self.proj.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode_fasta(self, fasta: str) -> torch.Tensor:
        _, _, toks = self.batch_converter([("p", fasta)])
        toks = toks.to(self.device)
        rep = self.model(toks, repr_layers=[30])["representations"][30]  # (1,L,640)
        mean1280 = rep[0].mean(0)  # (1280,)
        return self.proj(mean1280)  # (proj_dim,)

    @torch.no_grad()
    def encode_pdb(self, pdb_path: Path) -> torch.Tensor:
        """
        Encode a protein pocket PDB file into a fixed-size context vector.

        Args:
            pdb_path: Path to the protein pocket PDB file

        Returns:
            A 64-dimensional tensor representing the protein context
        """
        fasta = pocket_fasta(pdb_path)
        return self.encode_fasta(fasta)


# --- Test block ---
if __name__ == "__main__":
    from src.geoldm.my_ext.crossdock_dataset import get_dataloaders
    import torch

    # Load 2 samples from the dataloader
    loaders = get_dataloaders("crossdocked/crossdocked_pocket10", batch_size=2, subset=2)
    batch = next(iter(loaders["train"]))
    pdb_paths = batch["pdb_path"]  # list of strings

    encoder = ESM2PocketEncoder()
    pocket_vecs = []
    for pdb_path in pdb_paths:
        vec = encoder.encode_pdb(Path(pdb_path))
        print(f"Encoded {pdb_path}: shape {vec.shape}")
        print(vec)
        pocket_vecs.append(vec)
    pocket_vecs = torch.stack(pocket_vecs)
    print(f"Stacked pocket vectors shape: {pocket_vecs.shape}")
