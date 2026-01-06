# GPBench

GPBench is a benchmarking toolkit for genomic prediction. This repository reimplements and integrates many commonly used methods, including classic linear statistical approaches and machine learning / deep learning methods: rrBLUP, GBLUP, BayesA/B/C, SVR, Random Forest, XGBoost, LightGBM, DeepGS, DL_GWAS, DNNGP, SoyDNGP, DeepCCR, EIR, Cropformer, GEFormer, CropARNet, etc.

## Key Features
- Implements multiple genomic prediction methods and reproducible experimental workflows
- Supports GPU-accelerated deep learning methods (using PyTorch)
- Unified data loading and 10-fold cross-validation pipeline
- Outputs standardized evaluation metrics (PCC, MAE, MSE, R2) and per-fold predictions

## Important Structure
- `data/`: Example/real dataset directory, each species/dataset is a subfolder (e.g., `data/Cotton/`), containing:
	- `genetype.npz`: genotype matrix (typically saved as a NumPy array)
	- `phenotype.npz`: phenotype data (contains phenotype matrix and phenotype names)
- `method/`: subdirectories with implementations for each method (each method usually contains a main runner script plus hyperparameter/utility scripts)
- `result/`: default output directory for experimental results
- `environment.yml`: dependency file for creating a conda environment (recommended)

## Environment Setup (recommended: conda)
There is an `environment.yml` in the repository; it is recommended to create and activate a conda environment with it:

```bash
# On a machine with conda:
conda env create -f environment.yml
conda activate Benchmark
```

Notes:
- `environment.yml` contains most dependencies (including CUDA / cuDNN related packages and pip list) and is suitable for GPU-enabled environments (the file references CUDA 11.8 and matching RAPIDS/torch/cupy versions).
- Ensure the target machine has an NVIDIA driver compatible with CUDA 11.8/12.
- If you cannot use the environment file directly, you can install main dependencies into an existing Python environment as needed:

```bash
pip install -U numpy pandas scikit-learn torch torchvision optuna psutil xgboost lightgbm
```

(Warning: the above is a simplified installation; some packages may need additional configuration on GPU systems or certain platforms.)

## Data Format and Preparation
- Each species folder should contain `genetype.npz` and `phenotype.npz`.
- `genetype.npz` usually stores a 2D array (number of samples × number of SNPs).
- `phenotype.npz` typically includes two arrays: the phenotype matrix (number of samples × number of phenotypes) and a list of phenotype names.

Quickly view phenotype names for a dataset (e.g., `Cotton`):

```bash
python - <<'PY'
import numpy as np
obj = np.load('data/Cotton/phenotype.npz')
print(obj['arr_1'])
PY
```

## Quick Start (example with a method)
Most methods have a main script under `method/<Method>/`. Scripts usually accept parameters like `--methods`, `--species`, `--phe`, `--data_dir`, `--result_dir`, etc. Example:

```bash
# 1) Activate the environment
conda activate Benchmark

# 2) Run a single phenotype with DeepCCR (note: include trailing slash after --species)
python method/DeepCCR/DeepCCR.py \
	--methods DeepCCR/ \
	--species Cotton/ \
	--phe <PHENOTYPE_NAME> \
	--data_dir data/ \
	--result_dir result/
```

Common optional arguments (may vary across scripts):
- `--epoch`: number of training epochs (example scripts often default to 1000)
- `--batch_size`: batch size
- `--lr`: learning rate
- `--patience`: early stopping patience

You can inspect the argparse help for the specific script in the method directory:

```bash
python method/DeepCCR/DeepCCR.py -h
```

## Output Description
- Each run creates a directory under `result/` named by method/species/phenotype, e.g., `result/DeepCCR/Cotton/<PHENO>/`.
- Per-fold prediction results are typically saved as `fold{n}.csv`, containing `Y_test` and `Y_pred` columns.
- The script prints or saves average evaluation metrics at the end: PCC (Pearson correlation coefficient), MAE, MSE, R2, along with runtime and memory/GPU usage.

## Full Dataset Link
- [Species dataset](https://doi.org/10.6084/m9.figshare.31007608): contains genotype and phenotype data for 16 species.

## Running Tips & Troubleshooting
- For GPU usage, ensure `conda activate Benchmark` and that CUDA drivers are available; `torch.cuda.is_available()` should return True.
- If you encounter memory or GPU OOM issues, try reducing `--batch_size` or disabling some parallel settings in scripts.
- If running on CPU-only systems, some GPU-specific methods (RAPIDS or GPU-only implementations) may be unavailable or require alternative implementations.

## Contributing & Contact
- Contributions via issues and PRs are welcome. Please describe changes and testing in PRs.
- Contact: open an Issue in the repository or reach the repository owner (GitHub user: `xwzhang2118`).





