from src.geoldm.my_ext.crossdock_dataset import ATOM_TYPES, ATOM_TYPE_TO_IDX
import torch


# Create a function that returns probability for each node count
def n_nodes_distribution(n):
    """Returns probability for molecule with n nodes"""
    if n <= 0 or n >= 129:  # Out of range
        return 0.0
    if n < 20:
        return float(n) * 0.5  # Increasing probability for small molecules
    elif 20 <= n <= 50:
        return 10.0  # Higher probability in the middle range
    else:
        return max(0.1, 10.0 - (n - 50) * 0.2)  # Decreasing probability for large molecules


# Compute histogram by calling n_nodes_distribution for each integer
n_nodes_hist = torch.tensor([n_nodes_distribution(i) for i in range(129)], dtype=torch.float)
n_nodes_hist = n_nodes_hist / n_nodes_hist.sum()

crossdock_pocket10 = {
    "name": "crossdock_pocket10",
    "atom_encoder": ATOM_TYPE_TO_IDX,
    "atom_decoder": ATOM_TYPES,
    "max_n_nodes": 128,
    "with_h": False,
    "n_nodes": {
        int(i): float(p) for i, p in enumerate(n_nodes_hist.tolist())
    },  # dict for robust indexing
    "context_node_nf": 64,  # Each node receives a 64-dimensional context embedding
}
