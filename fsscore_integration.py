import torch
from rdkit import Chem
from fsscore.score import Scorer


class FSSynthEvaluator:
    """
    A class to integrate FSSCore functionality with AiZynthFinder.
    This allows evaluation of molecule synthesizability using the Focused Synthesizability Score.
    """

    def __init__(self, device=None, batch_size=32):
        """
        Initialize the FSSCore Scorer model.

        Args:
            device: The device to run the model on ('cuda' or 'cpu').
                   If None, will use CUDA if available.
            batch_size: Batch size for scoring molecules
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing FSSCore Scorer model on {self.device}...")
        self.model = Scorer(featurizer="graph_2D", batch_size=batch_size, verbose=False)
        print("FSSCore model loaded successfully!")

    def evaluate_smiles(self, smiles_list):
        """
        Evaluate a list of SMILES strings for synthesizability.

        Args:
            smiles_list: List of SMILES strings to evaluate

        Returns:
            Dictionary mapping each SMILES to its FSscore
        """
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        # Filter out invalid SMILES
        valid_smiles = []
        invalid_indices = []
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(smiles)
            else:
                invalid_indices.append(i)
                print(f"Warning: Invalid SMILES: {smiles}")

        if not valid_smiles:
            return {}

        # Get FSScores - returns a dict with keys 'mean' and 'std'
        scores_dict = self.model(valid_smiles)
        scores = scores_dict["mean"]

        # Create dictionary of results
        results = {smiles: float(score) for smiles, score in zip(valid_smiles, scores)}
        return results

    def evaluate_routes(self, finder, top_n=5):
        """
        Evaluate synthesis routes from AiZynthFinder results.

        Args:
            finder: AiZynthFinder instance with completed tree search
            top_n: Number of top routes to consider

        Returns:
            List of dictionaries with route info including FSscore
        """
        if not hasattr(finder, "routes") or not finder.routes:
            raise ValueError(
                "AiZynthFinder does not have any routes. Run tree_search() and build_routes() first."
            )

        routes_data = []
        for i, route in enumerate(finder.routes[:top_n]):
            target = route.target
            smiles = target.smiles
            score = self.evaluate_smiles(smiles)

            route_data = {
                "route_id": i,
                "smiles": smiles,
                "route_score": route.score,
                "fsscore": score.get(smiles, None),
                "num_steps": len(route.steps),
                "is_solved": route.is_solved,
            }
            routes_data.append(route_data)

        return routes_data


# Example usage
if __name__ == "__main__":
    # Initialize the FSSynthEvaluator
    fs_evaluator = FSSynthEvaluator()

    """     # Example SMILES to evaluate
    test_smiles = "Cc1cccc(c1N(CC(=O)Nc2ccc(cc2)c3ncon3)C(=O)C4CCS(=O)(=O)CC4)C"
    scores = fs_evaluator.evaluate_smiles(test_smiles)
    print(f"FSScore for test compound: {scores[test_smiles]:.4f}")

    # Try with a batch of molecules
    test_batch = [
        "Cc1cccc(c1N(CC(=O)Nc2ccc(cc2)c3ncon3)C(=O)C4CCS(=O)(=O)CC4)C",
        "O=C(Nc1ccc(cc1)c2ccccc2)c3ccccc3",
        "CC(C)(C)c1ccc(cc1)C(=O)Nc2ccc(C)c(Nc3nccc(n3)c4cccnc4)c2",
    ]

    batch_scores = fs_evaluator.evaluate_smiles(test_batch)
    print("\nBatch scoring results:")
    for smiles, score in batch_scores.items():
        print(f"{smiles}: {score:.4f}") """
