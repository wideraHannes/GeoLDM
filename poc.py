import torch
from qm9.models import get_latent_diffusion
from qm9 import dataset
from configs.datasets_config import get_dataset_info
from os.path import join
import pickle
from qm9 import visualizer as vis


def main():
    # Load dataset info
    dataset_info = get_dataset_info("qm9", remove_h=False)
    print("\nDataset info loaded")

    # Load pretrained model
    model_path = "outputs/qm9_latent2"  # Using qm9_latent2 model
    print(f"\nLoading model from: {model_path}")

    # Load model arguments
    with open(join(model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    # Force CPU usage in args
    args.cuda = False

    print(args)

    device = torch.device("cpu")
    print("Using device: cpu")

    # Create model
    model, nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, None
    )

    # Load pretrained weights
    model.load_state_dict(
        torch.load(join(model_path, "generative_model_ema.npy"), map_location="cpu")
    )
    model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Reduce number of diffusion steps for faster sampling
    original_steps = args.diffusion_steps
    args.diffusion_steps = 1  # Reduced from default (usually 1000)
    args.batch_size = 1
    model.diffusion_steps = args.diffusion_steps
    print(f"\nReduced diffusion steps from {original_steps} to {args.diffusion_steps}")

    # Load QM9 dataset
    print("\nLoading QM9 dataset...")
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    # Get a sample molecule (batch size 4)
    sample_batch = next(iter(dataloaders["train"]))
    print(len(sample_batch))
    print(f"\nsample_batch keys: {list(sample_batch.keys())}")
    x = sample_batch["positions"]
    h = {"categorical": sample_batch["one_hot"], "integer": sample_batch["charges"]}
    node_mask = sample_batch["atom_mask"]
    edge_mask = sample_batch["edge_mask"]
    context = None  # Not used for unconditional generation

    # Move to device
    x = x.to(device)
    h = {k: v.to(device) for k, v in h.items()}
    node_mask = node_mask.to(device).float().unsqueeze(-1)
    edge_mask = edge_mask.to(device).float()
    if context is not None:
        context = context.to(device)

    # Mask x and h with node_mask
    x = x * node_mask
    h["categorical"] = h["categorical"] * node_mask
    h["integer"] = h["integer"] * node_mask

    # Center x for each molecule using the node_mask
    masked_x = x * node_mask
    num_atoms = node_mask.sum(dim=1, keepdim=True)  # [batch, 1, 1]
    mean = masked_x.sum(dim=1, keepdim=True) / (num_atoms + 1e-8)
    x = x - mean  # Centered positions
    x = x * node_mask  # Mask again to ensure zeros where masked

    # Debug: check masking
    print("Unique values in node_mask:", torch.unique(node_mask.cpu()))
    print("Max abs(x * (1 - node_mask)):", (x * (1 - node_mask)).abs().max().item())

    print("\nStep 1: Prepare input and masks")
    # Step 1: Prepare input and masks
    # (already done above)

    print("\nStep 2: Sample from the diffusion model")
    # Step 2: Sample from the diffusion model
    with torch.no_grad():
        output_x, output_h = model.sample(
            n_samples=x.shape[0],
            n_nodes=x.shape[1],
            node_mask=node_mask,
            edge_mask=edge_mask,
            context=context,
        )
        print("Output x shape:", output_x.shape)
        print("Output h shape:", output_h["categorical"].shape)

    print("\nStep 3: Shape verification and visualization")
    # Step 3: Shape verification and visualization
    print(f"Input sample: {x.shape}")
    print(f"Output x: {output_x.shape}")
    print(f"Output h: {output_h['categorical'].shape}")

    # Visualize the decoded molecule using plot_data3d
    vis.plot_data3d(
        output_x.squeeze(0).cpu(),
        torch.argmax(output_h["categorical"].squeeze(0).cpu(), dim=1),
        dataset_info,
        spheres_3d=True,
    )

    # TODO: Implement molecule visualization
    print(
        "\nNote: To visualize molecules, implement the tensor_to_mol and "
        "visualize_mol functions"
    )
    print(
        "These would convert the tensor representations back to RDKit "
        "molecules and display them"
    )


if __name__ == "__main__":
    main()
