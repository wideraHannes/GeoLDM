## Pocket Conditioned GeoLDM

integrate pocket information into geoldm in multiple steps (crossdock 2020 dataset).
Dataset consists of pocket-ligand pairs.
The end goal is to provide solely the pocket at inference time and the model shall output binding ligands.
Read and optimize [Expose](https://pad.hhu.de/uhO1gxicS8S2RXKNxQmQkg?view)

1. using the conditional context slot already present

   - encoding pocket with ESM-2 (loosing orientation?) -> pocket_enc
   - passing pocket_enc into VAE & Diffusion as argument like they did in the og model with the properties
   - open questions:
     - Does this even work
     - how to train the model
     - does the VAE need to see the pocket?
     - possibility to use pretrained one?

2. implement a more sophisticated solution
   - Film Layer?
   - crossattention?
   - open questions:
     - how to maintain Equivariance i have heard attention would pose a problem!

### GeoLDM: Geometric Latent Diffusion Models for 3D Molecule Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MinkaiXu/GeoLDM/blob/main/LICENSE)
[![ArXiv](http://img.shields.io/badge/cs.LG-arXiv%3A2305.01140-B31B1B.svg)](https://arxiv.org/abs/2305.01140)

<!-- [[Code](https://github.com/MinkaiXu/GeoLDM)] -->

![cover](src/geoldm/equivariant_diffusion/framework.png)

Official code release for the paper "Geometric Latent Diffusion Models for 3D Molecule Generation", accepted at _International Conference on Machine Learning, 2023_.

### GeoLDM: background knowlege

**GeoLDM**  
Built on top of EGNN and EDM.

- Modified EGNN into a VAE  
  → _"EGNN as VAE"_

- uses EDM as diffusion model

### [EGNN](https://arxiv.org/pdf/2102.09844)

**Equivariant Graph Neural Networks**

- Equivariant to: rotation, translation, reflection, permutation
- No need for higher-order features
- Efficient and performant

---

### [EDM](https://arxiv.org/pdf/2203.17003)

**Equivariant Diffusion Model**

- For 3D molecule generation
- E(3)-equivariant network
- Learns to denoise diffusion
- Works on:
  - Atom coordinates (continuous)
  - Atom types (categorical)

## 🚀 Working with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and virtual environments.

### 🛠 1. Sync Environment

Install all dependencies and create the virtual environment:

```bash
uv sync
```

### 2. run scripts

Run Python scripts within the managed virtual environment:

```bash
uv run python your_script.py
```

or first activate the environment

```bash
source .venv/bin/activate
```

and then execute your file

```bash
uv run python your_script.py
```

### 3. Add Dependencies

Add new Packages with

```bash
uv add [package]
```

(this adds the package to `pyproject.toml` and `uv.lock`)

#### (optional) 4. Syncing after Pulling Changes

If someone else adds packages and you pull the changes, just run:

```bash
uv sync
```

It will install the correct versions based on the lockfile.

## Custom code Explanation

For now i put all custom code into the [my_ext](./src/geoldm/my_ext/) folder.
Here i implemented:

- a crossdock dataloader
- an execution [script](./src/geoldm/my_ext/main_crossdock.py) that wraps around the qm9 script to run training with my args.

so you can according to [2. run scripts]() execute this file with

```bash
uv run src/geoldm/my_ext/main_crossdock.py
```

### Dataset

We plan to train using the same data sets as [Pocket2Mol](https://github.com/pengxingang/Pocket2Mol) and [SBDD](https://github.com/luost26/3D-Generative-SBDD) model.

1. Download the dataset archive `crossdocked_pocket10.tar.gz` and the split file `split_by_name.pt` from [this link](https://drive.google.com/drive/folders/1CzwxmTpjbrt83z_wBzcQncq84OVDPurM).
2. Extract the TAR archive using the command: `tar -xzvf crossdocked_pocket10.tar.gz`.

select only a subset of folders and create a small training dataset.
