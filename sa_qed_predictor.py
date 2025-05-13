import torch
import torch.nn as nn
from rdkit.Chem import QED, Descriptors


class DrugLikeScorer(nn.Module):
    """
    A lightweight predictor for synthetic accessibility and drug-likeness scores.
    This module provides guidance signals to steer the diffusion model toward
    molecules that are more synthesizable and have better drug-likeness properties.

    Higher scores are better (inverted SA score since lower SA is better).
    """

    def __init__(self, sa_weight=1.0, qed_weight=0.3):
        super(DrugLikeScorer, self).__init__()
        self.sa_weight = sa_weight
        self.qed_weight = qed_weight

    def forward(self, mol):
        """
        Calculate synthetic accessibility and drug-likeness score for a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Combined score where higher is better
        """
        if mol is None:
            return torch.tensor(-10.0, device=self._get_device())

        try:
            sa_score = Descriptors.SAscore(mol)  # Lower is better (1-10 scale)
            qed_score = QED.qed(mol)  # Higher is better (0-1 scale)

            # Invert SA score so higher is better
            sa_norm = -sa_score  # Now higher is better

            # Combine scores (higher is better)
            combined_score = self.sa_weight * sa_norm + self.qed_weight * qed_score

            return torch.tensor(combined_score, device=self._get_device())
        except Exception:
            # Return a very low score if calculation fails
            return torch.tensor(-10.0, device=self._get_device())

    def _get_device(self):
        """Helper to get the current device."""
        return "cuda" if torch.cuda.is_available() else "cpu"
