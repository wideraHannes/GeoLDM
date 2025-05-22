# GeoLDM Project Restructuring Guide

This guide outlines steps to reorganize the GeoLDM project according to Python package best practices, optimized for the `uv` package manager and ensuring proper absolute imports throughout.

## 1. Project Structure Reorganization

### 1.1 Create proper package structure

```
GeoLDM/
├── src/
│   └── geoldm/
│       ├── __init__.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   └── ... (analysis modules)
│       ├── configs/
│       │   ├── __init__.py
│       │   └── ... (config modules)
│       ├── data/
│       │   ├── __init__.py
│       │   └── ... (dataset modules)
│       ├── egnn/
│       │   ├── __init__.py
│       │   └── ... (EGNN modules)
│       ├── equivariant_diffusion/
│       │   ├── __init__.py
│       │   └── ... (diffusion modules)
│       ├── models/
│       │   ├── __init__.py
│       │   └── ... (model definitions)
│       ├── qm9/
│       │   ├── __init__.py
│       │   └── ... (QM9-specific modules)
│       ├── my_ext/
│       │   ├── __init__.py
│       │   └── ... (extensions)
│       └── utils/
│           ├── __init__.py
│           └── ... (utility modules)
├── scripts/
│   ├── analyze_samples.py
│   ├── eval_conditional_qm9.py
│   ├── eval_sample.py
│   ├── main_qm9.py
│   └── ... (other entry point scripts)
├── tests/
│   ├── test_analyze_and_save.py
│   ├── test_sampling.py
│   └── ... (other test files)
├── notebooks/
│   ├── test_pocket_dataset.ipynb
│   └── ... (other notebooks)
├── pyproject.toml
├── README.md
└── .gitignore
```

## 2. Package Configuration with pyproject.toml

### 2.1 Update pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "geoldm"
version = "0.1.0"
description = "Geometric Latent Diffusion Models for 3D Molecule Generation"
readme = "README.md"
requires-python = ">=3.8"
license = {file = "LICENSE"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
dependencies = [
    "torch>=1.9.0",
    "numpy>=1.20.0",
    "matplotlib>=3.4.0",
    "rdkit>=2021.03.1",
    "wandb>=0.12.0",
    "imageio>=2.9.0",
    # Add other dependencies from your requirements.txt
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.1.0",
    "ruff>=0.0.54",
]

[tool.hatch.build.targets.wheel]
packages = ["src/geoldm"]

[tool.ruff]
line-length = 100
target-version = "py38"

[tool.black]
line-length = 100
target-version = ["py38"]
```

## 3. Import Refactoring

### 3.1 Create proper __init__.py files

Create empty `__init__.py` files for all subpackages to make Python imports work correctly.

### 3.2 Update imports in all modules

Replace relative imports with absolute imports throughout the codebase:

```python
# Old import style (often with sys.path manipulation):
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from qm9.utils import some_function

# New import style (absolute imports):
from geoldm.qm9.utils import some_function
```

## 4. Entry Point Scripts

Convert main script files to use the installed package:

```python
#!/usr/bin/env python
# scripts/main_qm9.py

import argparse
from geoldm.models import create_model
from geoldm.qm9.dataset import get_dataloaders
from geoldm.utils import setup_training

def main():
    parser = argparse.ArgumentParser()
    # Add argument parsing...
    args = parser.parse_args()
    
    # Use absolute imports from the installed package
    model = create_model(args)
    # Rest of the code...

if __name__ == "__main__":
    main()
```

## 5. Environment and uv Setup

### 5.1 Set up uv with pyproject.toml

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with uv
uv venv

# Install package in development mode
uv pip install -e ".[dev]"
```

### 5.2 Create uv.lock file (if needed)

```bash
uv pip freeze > uv.lock
```

## 6. Implementation Steps

1. **Create the directory structure** first, making subdirectories under `src/geoldm/`.
2. **Move existing code** into the new structure, preserving file paths inside each module.
3. **Create `__init__.py` files** in each directory to make proper imports work.
4. **Update imports** in all files to use absolute imports from `geoldm` package.
5. **Move scripts** to the `scripts/` directory and update their imports.
6. **Update tests** to use the new import structure.
7. **Configure pyproject.toml** according to the example above.
8. **Test installation** with `uv pip install -e .`

## 7. Common Import Patterns

### 7.1 Expose key functions in __init__.py

In module `__init__.py` files, expose important functions to simplify imports:

```python
# src/geoldm/qm9/__init__.py
from .dataset import get_dataloaders, QM9Dataset
from .utils import prepare_context_pocket

# This allows users to import directly:
# from geoldm.qm9 import get_dataloaders
```

### 7.2 Test imports

Update test files to use the installed package:

```python
# tests/test_sampling.py
import torch
from geoldm.models import SamplingModel
from geoldm.qm9.utils import prepare_context

def test_sampling():
    # Test code...
```

## 8. Data Files Handling

Ensure data files are properly packaged:

```python
# Access data files with appropriate paths
from importlib import resources
import geoldm.configs

def get_config_path():
    with resources.path("geoldm.configs", "qm9_config.yaml") as path:
        return path
```

## 9. Final Testing

After implementing all changes:

1. **Run tests** to ensure all imports work correctly
2. **Try running main scripts** from any directory to verify imports
3. **Check installed package** can be imported in a Python REPL
