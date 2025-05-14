from my_ext.crossdock_dataset import ATOM_TYPES, ATOM_TYPE_TO_IDX
import torch

# Create a dummy histogram for n_nodes - we'll assume molecules have sizes from 1-128 nodes
# with a higher probability for sizes in the middle range (20-50 nodes)
n_nodes_hist = torch.zeros(129)  # 0-128 nodes
for i in range(1, 129):
    if i < 20:
        n_nodes_hist[i] = i * 0.5  # Increasing probability for small molecules
    elif 20 <= i <= 50:
        n_nodes_hist[i] = 10.0  # Higher probability in the middle range
    else:
        n_nodes_hist[i] = max(
            0.1, 10.0 - (i - 50) * 0.2
        )  # Decreasing probability for large molecules

# Normalize to make it a proper distribution
n_nodes_hist = n_nodes_hist / n_nodes_hist.sum()

crossdock_pocket10 = {
    "name": "crossdock_pocket10",
    "atom_encoder": ATOM_TYPE_TO_IDX,
    "atom_decoder": ATOM_TYPES,
    "max_n_nodes": 128 + 512,  # lig + pocket
    "with_h": False,
    "n_nodes": n_nodes_hist,  # Add the node distribution histogram
}
