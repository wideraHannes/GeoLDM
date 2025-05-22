import random
from pathlib import Path
from rdkit import Chem
from collections import defaultdict


def count_atoms_in_dataset(dataset_path, folder_percentage=100):
    atom_counts = defaultdict(int)
    all_folders = [f for f in Path(dataset_path).iterdir() if f.is_dir()]

    # Select a subset of folders based on the percentage
    num_folders = max(1, int(len(all_folders) * (folder_percentage / 100)))
    selected_folders = random.sample(all_folders, num_folders)

    print(
        f"Selected {len(selected_folders)} out of {len(all_folders)} folders ({folder_percentage}%)"
    )

    sdf_files = []
    for folder in selected_folders:
        sdf_files.extend(folder.rglob("*.sdf"))

    print(f"Found {len(sdf_files)} SDF files in the selected folders")

    for sdf_file in sdf_files:
        try:
            supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            for mol in supplier:
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    atom_counts[atom.GetSymbol()] += 1
        except Exception as e:
            print(f"Error processing {sdf_file}: {e}")

    return atom_counts


def main():
    dataset_path = "crossdocked/crossdocked_pocket10"
    folder_percentage = 100  # Adjust this percentage as needed
    atom_counts = count_atoms_in_dataset(dataset_path, folder_percentage=folder_percentage)

    print("\n--- Atom Counts ---")
    for atom, count in sorted(atom_counts.items()):
        print(f"{atom}: {count}")


if __name__ == "__main__":
    main()

    """output:
    
        Al: 4
            As: 2
            Au: 4
            B: 915
            Br: 7782
            C: 2966713
            Cl: 28039
            Co: 2
            Cr: 5
            Cu: 6
            F: 57054
            Fe: 460
            H: 8510
            Hg: 1
            I: 1649
            Li: 5
            Mg: 45
            Mo: 83
            N: 556175
            O: 685548
            P: 42099
            Ru: 53
            S: 45575
            Sc: 1
            Se: 55
            Si: 92
            Sn: 12
            V: 51
            W: 18
            Y: 2
                
    """
