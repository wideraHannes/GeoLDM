# my_ext/pocket_encoder.py
import os, sys, torch, torch.nn as nn
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[2].as_posix())  # repo root


# --------------------------------------------------------------------
# Pocket-feature MLP encoder
# --------------------------------------------------------------------
class PocketEncoder(nn.Module):
    """
    Simple MLP that pools pocket atoms (coords + one-hot + charge) → 64-d vector.
    """

    def __init__(self, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        inp = 3 + 28 + 1  # xyz + atom-type one-hot + charge
        self.mlp = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        pocket_pos: torch.Tensor,  # (B,N,3)
        pocket_feat: torch.Tensor,  # (B,N,29)  28+1
        pocket_mask: torch.Tensor,  # (B,N,1)   1 = real atom
    ) -> torch.Tensor:  # (B,64)
        """
        Encode pocket atoms into a fixed 64-dimensional vector
        
        Args:
            pocket_pos: Pocket atom positions, shape (B,N,3)
            pocket_feat: Pocket atom features (one-hot + charge), shape (B,N,29)
            pocket_mask: Binary mask, 1 = real atom, shape (B,N,1)
            
        Returns:
            A 64-dimensional context vector for each batch sample, shape (B,64)
        """
        B, N, _ = pocket_pos.shape
        # concat features along last dim
        h = torch.cat([pocket_pos, pocket_feat], dim=-1)  # (B,N,32)
        h = self.mlp(h)  # (B,N,64)
        h = h * pocket_mask  # zero padding
        z = h.sum(dim=1) / (pocket_mask.sum(1) + 1e-8)  # (B,64)
        return z


# --------------------------------------------------------------------
# Smoke-test
# --------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from crossdock_dataset import get_dataloaders

    root = sys.argv[1] if len(sys.argv) > 1 else "crossdocked/crossdocked_pocket10"
    loaders = get_dataloaders(
        root_dir=root,
        batch_size=2,
        subset=4,  # tiny dev split
        max_ligand_atoms=128,
        max_pocket_atoms=512,
    )

    batch = next(iter(loaders["train"]))
    # --- slice pocket tensors in batch shape ------------------------
    pocket_mask = batch["pocket_mask"]  # (B,N,1)
    pocket_pos = batch["positions"] * pocket_mask  # (B,N,3)
    pocket_feat = torch.cat([batch["x"], batch["charges"]], dim=-1) * pocket_mask  # (B,N,29)

    enc = PocketEncoder()
    z = enc(pocket_pos, pocket_feat, pocket_mask)  # (B,64)

    print(z[0])

    print("Pocket latent shape :", tuple(z.shape))
    print("mean, std           :", z.mean().item(), z.std().item())
