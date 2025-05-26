import shutil
from pathlib import Path
from rdkit import Chem

from configs.paths import CROSSDOCKED_DATA

# QM9 dataset only contains these atom types
ATOM_TYPES = ["H", "C", "N", "O", "F"]


def create_filtered_dataset(source_path, target_path, allowed_atom_types=ATOM_TYPES):
    """
    Create a new dataset with only molecules that contain atoms exclusively from the allowed_atom_types list.
    If a subfolder contains valid molecules, all files in that subfolder are copied.

    Args:
        source_path: Path to the source dataset
        target_path: Path to create the target dataset
        allowed_atom_types: List of allowed atom types (default: QM9 atom types)
    """
    # Create target directory if it doesn't exist
    source_path = Path(source_path)
    target_path = Path(target_path)
    if not target_path.exists():
        target_path.mkdir(parents=True)
        print(f"Created target directory: {target_path}")

    all_folders = [f for f in source_path.iterdir() if f.is_dir()]
    print(f"Found {len(all_folders)} folders in {source_path}")

    # Track statistics
    total_molecules = 0
    filtered_molecules = 0
    copied_subfolders = 0
    copied_files = 0
    molecules_in_5atom_dataset = 0

    for main_folder in all_folders:
        subfiles_sdf = list(main_folder.glob("*.sdf"))
        subfolder_valid = True

        for sdf_file in subfiles_sdf:
            try:
                supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)

                for mol in supplier:
                    total_molecules += 1
                    if mol is None:
                        continue

                    # Check if any atom is not in allowed types
                    for atom in mol.GetAtoms():
                        if atom.GetSymbol() not in allowed_atom_types:
                            subfolder_valid = False
                            filtered_molecules += 1
                            break

                    if not subfolder_valid:
                        break

                if not subfolder_valid:
                    break

            except Exception as e:
                print(f"Error processing {sdf_file}: {e}")
                subfolder_valid = False
                break

        # If all molecules are valid, copy the entire subfolder
        if subfolder_valid and subfiles_sdf:
            copied_subfolders += 1

            # Create target directory
            relative_path = main_folder.relative_to(source_path)
            target_subfolder = target_path / relative_path
            target_subfolder.mkdir(parents=True, exist_ok=True)

            # Copy all files from subfolder and count molecules
            for file in main_folder.glob("*.*"):
                target_file = target_subfolder / file.name
                shutil.copy2(file, target_file)
                copied_files += 1

                # Count molecules in copied SDF files
                if file.suffix.lower() == ".sdf":
                    try:
                        mol_supplier = Chem.SDMolSupplier(str(file), removeHs=False, sanitize=False)
                        molecules_in_5atom_dataset += sum(
                            1 for mol in mol_supplier if mol is not None
                        )
                    except Exception as e:
                        print(f"Error counting molecules in {file}: {e}")

    print("\n--- Filtering Summary ---")
    print(f"Original dataset: {total_molecules} molecules, {len(all_folders)} subfolders")
    print(f"Filtered: {filtered_molecules} molecules (non-QM9 atom types)")
    print(f"Result: {copied_subfolders} valid subfolders, {copied_files} files copied")
    print(f"5-atom dataset: {molecules_in_5atom_dataset} molecules")
    if total_molecules > 0:
        print(
            f"Reduction: {100 - (molecules_in_5atom_dataset / total_molecules * 100):.2f}% fewer molecules"
        )


def main():
    source_path = CROSSDOCKED_DATA / "crossdocked_pocket10"
    target_path = CROSSDOCKED_DATA / "crossdocked_5atoms"

    print(f"Creating filtered dataset with QM9 atom types {ATOM_TYPES}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")

    create_filtered_dataset(source_path, target_path)


if __name__ == "__main__":
    main()
