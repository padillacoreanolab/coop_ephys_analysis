# Cooperation Electrophysiology Analysis

This repository contains local and HiPerGator workflows for analyzing cooperation versus selfish behavior in mice, with a focus on behavior extraction, LFP analysis, and behavior-ephys integration.

## Project Overview

This project supports a hybrid analysis workflow:

- Local workflows for behavior extraction, behavior preprocessing, and smaller debugging tasks
- HiPerGator workflows for LFP analysis, batch processing, and larger computational jobs
- Integration workflows for aligning behavioral events with electrophysiology data

## Repository Structure

```text
coop_ephys_analysis/
|-- external/                     # External lab repositories or Git submodules
|-- src/coop_ephys_analysis/      # Reusable importable Python package code
|-- scripts/local/                # Scripts intended to run locally
|-- scripts/hpg/                  # Scripts intended to run on HiPerGator
|-- configs/                      # Configuration files for paths and analysis parameters
|-- notebooks/                    # Exploratory analysis and visualization notebooks
|-- data/                         # Raw and processed data, ignored by Git
|-- results/                      # Generated outputs, ignored by Git
`-- docs/                         # Setup notes, migration notes, and project documentation
```

## External Lab Repository

This project uses the original lab repository `diff_fam_social_memory_ephys` as a Git submodule:

```text
external/diff_fam_social_memory_ephys/
```

Treat this folder as external lab code. Do not directly edit files inside it unless you intentionally want to modify the original lab repository. Project-specific wrappers, cleaned pipeline code, and adaptations should live in this repository's own `src/coop_ephys_analysis/` package.

## Cloning This Repository

Because this repository uses a Git submodule, clone it with:

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/coop_ephys_analysis.git
```

If you already cloned it without the submodule, run:

```bash
git submodule update --init --recursive
```

To update the external lab repository to its latest version:

```bash
cd external/diff_fam_social_memory_ephys
git pull origin main
cd ../..
git add external/diff_fam_social_memory_ephys
git commit -m "Update diff_fam_social_memory_ephys submodule"
git push
```

## Environment Files

This repo keeps the base analysis environment separate from optional GPU dependencies:

- `environment.yml`: the recommended Conda/Mamba environment for CPU-safe local and HiPerGator work.
- `requirements.txt`: the same base Python package set in pip format, useful for quick checks or pip-only installs.
- `requirements-gpu.txt`: optional GPU acceleration packages for local machines that need CUDA runtime wheels.

The split is intentional. Most behavior extraction, preprocessing, event alignment, plotting, and non-GPU LFP work should run with only `environment.yml` or `requirements.txt`. GPU packages are larger, more platform-specific, and can conflict with cluster CUDA module systems, so they are opt-in.

## Base Environment Setup

Use Mamba when possible because it resolves scientific Python environments faster than Conda:

```bash
mamba env create -f environment.yml
mamba activate coop_ephys_env
```

or, with Conda:

```bash
conda env create -f environment.yml
conda activate coop_ephys_env
```

If the environment already exists and `environment.yml` has changed, update it with:

```bash
mamba env update -n coop_ephys_env -f environment.yml --prune
```

or:

```bash
conda env update -n coop_ephys_env -f environment.yml --prune
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name coop_ephys_env --display-name "Python (coop_ephys_env)"
```

Then select `Python (coop_ephys_env)` inside Jupyter Notebook, JupyterLab, or VS Code.

## NumPy And SpikeInterface Pin

`numpy` is pinned to `1.26.4` because the migrated LFP pipeline currently depends on `spikeinterface==0.100.6`. That SpikeInterface version includes compiled extensions built against NumPy 1.x and can crash the Python or Jupyter kernel with NumPy 2.x.

Do not upgrade NumPy to 2.x unless SpikeInterface is also upgraded to a NumPy-2-compatible release and the LFP tests/notebooks are rechecked.

## Optional GPU Setup

GPU acceleration is mainly useful for LFP spectral analyses, especially calculations that use `spectral_connectivity`. The base environment does not install GPU packages automatically.

On a local Windows/Linux machine with an NVIDIA GPU and no cluster CUDA module system, activate the base environment and then run:

```bash
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` includes `cupy-cuda12x==13.6.0` because CuPy 14+ requires NumPy 2.x, while this project currently stays on NumPy 1.x for `spikeinterface==0.100.6` compatibility.

To enable GPU use in notebooks or scripts, set:

```bash
SPECTRAL_CONNECTIVITY_ENABLE_GPU=true
```

In a notebook, this can be done with:

```python
%env SPECTRAL_CONNECTIVITY_ENABLE_GPU=true
```

On Windows, you can also add `SPECTRAL_CONNECTIVITY_ENABLE_GPU=true` as a user environment variable.

## HiPerGator GPU Workflow

On HiPerGator, prefer the base Conda/Mamba environment plus CUDA modules provided by the cluster. Do not install the local `nvidia-*-cu12` runtime wheels from `requirements-gpu.txt` unless you have a specific reason; the cluster module system normally provides CUDA.

Typical workflow:

1. Open HiPerGator Open OnDemand.
2. Start an interactive JupyterLab session using a GPU partition.
3. Request a GPU through the Generic Resource Request field, for example `gpu:geforce:1` or `gpu:a100:1`, depending on the hardware needed.
4. Clone this repository with submodules, or pull the latest version if it is already present.
5. Load Conda/Mamba and CUDA modules.
6. Create or activate `coop_ephys_env`.
7. Register the environment as a Jupyter kernel if needed.
8. Select the `Python (coop_ephys_env)` kernel in JupyterLab.

Example commands after the Jupyter session starts:

```bash
module load conda
module load cuda/12.4.1

git clone --recurse-submodules https://github.com/YOUR_USERNAME/coop_ephys_analysis.git
cd coop_ephys_analysis

mamba env create -f environment.yml
mamba activate coop_ephys_env
python -m ipykernel install --user --name coop_ephys_env --display-name "Python (coop_ephys_env)"
```

If the environment already exists:

```bash
module load conda
module load cuda/12.4.1
cd coop_ephys_analysis
mamba activate coop_ephys_env
mamba env update -n coop_ephys_env -f environment.yml --prune
```

To check that CuPy can see the GPU after CUDA is loaded:

```bash
python -c "import cupy as cp; x = cp.array([1, 2, 3]); print(cp.linalg.norm(x))"
```

To check the core LFP imports:

```bash
python -c "import numpy, h5py, bidict, spikeinterface, spectral_connectivity; print('LFP imports OK')"
```

## Notes For Pipeline Migration

The cleaned behavior extraction code already lives under `src/coop_ephys_analysis/behavior/`. Migrated LFP code should follow the same pattern and live under `src/coop_ephys_analysis/lfp/`, while notebooks should remain lightweight examples or analysis entry points.
