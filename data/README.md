# Data

The repository includes only compact synthetic inputs that can be redistributed with the code.

## Included

| File | Manuscript experiment |
|---|---|
| `synthetic/simulation_gene_data.csv` | Gene, 2D |
| `synthetic/dyngen.csv` | Dyngen, 5D |

Generate `synthetic/gaussian_1000d.csv` with `scripts/generate_gaussian.py`.

## Expected external processed files

Place the following author-preprocessed matrices under `data/real/`:

| Filename | Shape described in the manuscript | Config |
|---|---:|---|
| `emt_10d.csv` | 3,132 cells × 10 dimensions | `configs/emt.yaml` |
| `eb_pca50.csv` | 16,819 cells × 50 dimensions | `configs/eb_5d.yaml`, `configs/eb_50d.yaml` |
| `eb_pca100.csv` | 16,819 cells × 100 dimensions | `configs/eb_100d.yaml` |
| `cite_pca50.csv` | 31,240 cells × 50 dimensions | `configs/cite.yaml` |
| `mouse_pca50.csv` | 49,302 cells × 50 dimensions | `configs/mouse.yaml` |

All files must have a `samples` column followed by numeric feature columns. The configs select the reported number of leading components.

The large perturbation benchmark is generated with `scripts/generate_perturbation.py`. Keep manuscript-scale outputs under `data/perturbation/`; that directory is ignored because a 5000-condition cell table is too large for normal GitHub storage.
