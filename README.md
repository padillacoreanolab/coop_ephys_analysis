# Cooperation Electrophysiology Analysis

This repository contains local and HiPerGator-based workflows for analyzing cooperation versus selfish behavior in mice, with a focus on behavior extraction, LFP analysis, and behavior-ephys integration.

## Project Overview

This project supports a hybrid analysis workflow:

- Local workflows for behavior extraction, behavior preprocessing, and smaller debugging tasks
- HiPerGator workflows for LFP analysis, batch processing, and larger computational jobs
- Integration workflows for aligning behavioral events with electrophysiology data

## Repository Structure

```text
coop_ephys_analysis/
├── external/                     # External lab repositories or Git submodules
├── src/coop_ephys_analysis/      # Reusable importable Python package code
├── scripts/local/                # Scripts intended to run locally
├── scripts/hpg/                  # Scripts intended to run on HiPerGator
├── configs/                      # Configuration files for paths and analysis parameters
├── notebooks/                    # Exploratory analysis and visualization notebooks
├── data/                         # Raw and processed data, ignored by Git
├── results/                      # Generated outputs, ignored by Git
└── docs/                         # Setup notes, migration notes, and project documentation
```
## External Lab Repository

This project uses the original lab repository `diff_fam_social_memory_ephys` as a Git submodule:

```text
external/diff_fam_social_memory_ephys/
```

This submodule should be treated as external lab code. Do not directly edit files inside this folder unless you intentionally want to modify the original lab repository. Project-specific wrappers or adaptations should be written in this repository’s own source code instead.

## Cloning This Repository

Because this repository uses a Git submodule, clone it with:

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/coop_ephys_analysis.git
```

If you already cloned it without the submodule, run:

```bash
git submodule update --init --recursive
```

## Updating the External Lab Submodule

To update the external lab repository to its latest version:

```bash
cd external/diff_fam_social_memory_ephys
git pull origin main
cd ../..
git add external/diff_fam_social_memory_ephys
git commit -m "Update diff_fam_social_memory_ephys submodule"
git push
```

## Environment Setup

This project uses a Conda/Mamba environment defined in:

```text
environment.yml
```

To create the environment, run:

```bash
mamba env create -f environment.yml
```

or, if using Conda:

```bash
conda env create -f environment.yml
```

To activate the environment:

```bash
mamba activate coop_ephys_env
```

or:

```bash
conda activate coop_ephys_env
```

To register this environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name coop_ephys_env --display-name "Python (coop_ephys_env)"
```

After this, select the kernel named:

```text
Python (coop_ephys_env)
```

inside Jupyter Notebook or VS Code.

If the environment already exists and `environment.yml` has been updated, update the environment with:

```bash
mamba env update -f environment.yml --prune
```

or:

```bash
conda env update -f environment.yml --prune
```