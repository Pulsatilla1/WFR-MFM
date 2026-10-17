# WFR-MFM

Clean reference implementation for **WFR-MFM: One-Step Inference for Dynamic Unbalanced Optimal Transport**.

WFR-MFM learns a mean velocity field and a mean mass-growth field over finite time intervals. A trained model advances positions and masses directly,

\[
x_T = x_t + (T-t)v_\theta(x_t,t,T), \qquad
m_T = m_t\exp\{(T-t)h_\phi(x_t,t,T)\},
\]

so inference needs one network evaluation per interval. The same checkpoint can use more substeps when a finer speed–accuracy trade-off is desired.

## What is included

- A compact PyTorch implementation of Algorithms 1, 2, and 4 from the manuscript.
- Full-batch and mini-batch WFR optimal entropy transport (OET) couplings.
- Unconditional and perturbation-conditioned WFR-MFM models.
- Training, inference, and evaluation commands.
- Manuscript hyperparameter files for Gene, Dyngen, Gaussian, EMT, EB, CITE-seq, Mouse hematopoiesis, and the perturbation experiment.
- Small synthetic Gene and Dyngen datasets, plus generators for the 1000D Gaussian and perturbation benchmarks.

The repository intentionally excludes baseline implementations, exploratory notebooks, intermediate checkpoints, repeated script variants, manuscript artwork, and cached result files. See [docs/PAPER_CODE_MAP.md](docs/PAPER_CODE_MAP.md) and [docs/RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md).

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e '.[dev]'
pytest
```

PyTorch should be installed with the CUDA build appropriate for the local system when GPU training is required.

## Quick start

Train the small Gene experiment:

```bash
python scripts/train.py --config configs/gene.yaml --output outputs/gene/model.pt
```

Run one-step inference per observed interval:

```bash
python scripts/infer.py \
  --checkpoint outputs/gene/model.pt \
  --data data/synthetic/simulation_gene_data.csv \
  --output outputs/gene/predictions.csv
```

Evaluate weighted \(W_1\) and relative mass error (RME):

```bash
python scripts/evaluate.py \
  --checkpoint outputs/gene/model.pt \
  --data data/synthetic/simulation_gene_data.csv \
  --output outputs/gene/metrics.csv
```

Use `--steps-per-interval 10` in either inference or evaluation to reproduce the multi-step variant without retraining.

Benchmark the speed–accuracy trade-off used in Table 1 and Figure 2:

```bash
python scripts/benchmark.py \
  --checkpoint outputs/gene/model.pt \
  --data data/synthetic/simulation_gene_data.csv \
  --steps 1 2 5 10 20 30 \
  --runs 1000 \
  --output outputs/gene/benchmark.csv
```

## Paper experiments

Each YAML file records the values stated in the manuscript. For example:

```bash
python scripts/train.py --config configs/dyngen.yaml --output outputs/dyngen/model.pt
python scripts/train.py --config configs/gaussian.yaml --output outputs/gaussian/model.pt
python scripts/train.py --config configs/emt.yaml --output outputs/emt/model.pt
```

Generate the Gaussian benchmark before training:

```bash
python scripts/generate_gaussian.py --output data/synthetic/gaussian_1000d.csv
```

Large processed biological matrices are not duplicated in this GitHub folder. Place them under `data/real/` using the filenames in [data/README.md](data/README.md).

## Perturbation-conditioned model

Generate a small smoke-test dataset:

```bash
python scripts/generate_perturbation.py \
  --conditions 8 --cells 300 --duration 2 \
  --data-output data/synthetic/perturbation_small.csv \
  --embedding-output data/synthetic/perturbation_small_embeddings.csv
```

For the manuscript-scale experiment, generate 100 training conditions and 5000 held-out conditions with the same simulator settings, then train:

```bash
python scripts/train_conditional.py \
  --config configs/perturbation.yaml \
  --output outputs/perturbation/model.pt
```

Conditional inference accepts any row from the condition-embedding table:

```bash
python scripts/infer_conditional.py \
  --checkpoint outputs/perturbation/model.pt \
  --control data/perturbation/control.csv \
  --embeddings data/perturbation/test_embeddings.csv \
  --conditions 1 2 3 \
  --output outputs/perturbation/predictions.csv
```

## Input format

Unconditional data use one row per cell:

```text
samples,x1,x2,...
0, ...
1, ...
```

Conditional data use `samples=0` for the shared control and positive integer labels for perturbations. The embedding table uses the same `samples` labels and one numeric column per condition variable.

## Reproducibility notes

- Default networks use five hidden layers, 256 units per layer, and LeakyReLU activations as reported in the manuscript.
- OET plans are computed once before neural-network optimization.
- The traveling-Gaussian bandwidth is configurable; manuscript configs use the zero-bandwidth limit found in the source experiments.
- The training target uses the directional derivative identity with a stopped-gradient target.
- Mini-batch OET partitions source and target cells into paired blocks. This is the scalable approximation used for large datasets.
- Random seeds, feature order, time points, normalization statistics, and the full YAML configuration are stored in each checkpoint.
- Conditional embedding statistics are fitted on the complete training embedding table, including its control row, as in the source experiment.

## Citation

If this code supports your work, cite the accompanying manuscript:

> X. Wang, R. Wang, Q. Peng, P. Zhou, and T. Li. WFR-MFM: One-Step Inference for Dynamic Unbalanced Optimal Transport. 2026.

Repository licensing should be selected by the authors before public release.
