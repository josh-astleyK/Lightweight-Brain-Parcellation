# Lightweight Brain Parcellation

Whole-brain MRI parcellation into 134 regions using lightweight UMamba models. Trained on a multi-site T1-weighted dataset with expert-labelled parcellations.

## Models

| Model | Params | Mean Dice | Notes |
|---|---|---|---|
| `umamba_base` | 44 M | 0.841 | Full-width baseline |
| `umamba_fwr` | 11 M | 0.825 | Filter-width reduced; recommended |
| `umamba_lite_int8` | 11 M | 0.814 | INT8 quantized; Linux x86 only |

All models output a NIfTI label map with 134 regions (see `assets/label_map.json`).

## Requirements

- Python 3.11
- PyTorch ≥ 2.1
- Supported platforms: **Linux x86_64** (CUDA GPU recommended; CPU supported) and **Windows x86_64** (CPU only)
- macOS is not supported (mamba-ssm requires triton/CUDA extensions unavailable on ARM64)

## Installation

```bash
conda env create -f environment.yml
conda activate brain-parc
```

Then install `mamba-ssm` for your platform.

**Linux with CUDA:**
```bash
pip install causal-conv1d mamba-ssm
```

**Linux CPU-only:**
```bash
MAMBA_SKIP_CUDA_BUILD=TRUE CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE \
MAMBA_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE \
pip install --no-build-isolation causal-conv1d==1.4.0
MAMBA_SKIP_CUDA_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
pip install --no-build-isolation --no-deps mamba-ssm==2.2.2
python mamba_cpu_installer.py
```

**Windows (CPU only):**
```bat
set MAMBA_SKIP_CUDA_BUILD=TRUE
set CAUSAL_CONV1D_SKIP_CUDA_BUILD=TRUE
set MAMBA_FORCE_BUILD=TRUE
set CAUSAL_CONV1D_FORCE_BUILD=TRUE
pip install --no-build-isolation causal-conv1d==1.4.0
pip install --no-build-isolation --no-deps mamba-ssm==2.2.2
python mamba_cpu_installer.py
```

`mamba_cpu_installer.py` patches the installed package to remove dependencies on triton and CUDA extensions not needed for CPU inference.

## Weights

Model weights are stored with Git LFS.  After cloning, pull them with:
```bash
git lfs pull
```

Or download individual files from the [Releases](../../releases) page and place them in `models/weights/`.

## Usage

```bash
# single image
python parcellate.py --input t1.nii.gz --model umamba_fwr

# specify output path
python parcellate.py --input t1.nii.gz --model umamba_fwr --output sub01_parc.nii.gz

# batch (directory)
python parcellate.py --input scans/ --model umamba_fwr --output parcellations/

# GPU inference
python parcellate.py --input t1.nii.gz --model umamba_base --device cuda

# limit CPU threads
python parcellate.py --input t1.nii.gz --model umamba_fwr --threads 8
```

Output is saved as `{input_stem}_{model}_parcellation.nii.gz` in the same directory as the input unless `--output` is specified.

## Label map

`assets/label_map.json` maps integer labels (0–133) to anatomical region names following the Neuromorphometrics protocol.  Label 0 is background.

## Architecture

All three models share the same UMambaBot backbone: a 6-stage residual U-Net encoder with a Mamba SSM bottleneck (channel-token mode at 5³ spatial resolution) and a symmetric residual decoder.

Architecture code is in `models/architecture.py`.

## Citation

If you use these models, please cite:

```bibtex
@inproceedings{astley2026rapid,
  title     = {Rapid Whole-Brain Parcellation},
  author    = {Astley, Joshua R. and Rodionov, Roman and Primidis, Thomas and Singh, Pushpendra and Li, Kuo and Duncan, John S. and Sparks, Rachel},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Satellite Events, Workshop on Machine Learning in Clinical Neuroimaging (MLCN)},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  note      = {In press; DOI to be added}
}
```

## Licence

Code: Apache 2.0.  Model weights: Apache 2.0.  See [LICENSE](LICENSE).
