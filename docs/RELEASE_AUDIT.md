# Release audit

This folder was reconstructed from the manuscript and the research workspace. The original workspace is unchanged.

## Retained

- WFR-MFM model architecture.
- WFR OET coupling, traveling-Dirac path, mean-flow objective, training, and inference.
- Conditional perturbation-response extension used in the manuscript.
- Evaluation metrics and manuscript hyperparameters.
- Compact synthetic data and deterministic data generators.

## Excluded from the clean release

- Repeated dated script variants and unnamed scratch scripts.
- WFR-FM and DeepRUOT baseline code that is not required by WFR-MFM.
- The separate U-Pert inverse-design project and its DEG/cell-type design workflows.
- Adobe Illustrator sources, manuscript figures, spreadsheets, and plotting experiments.
- Training logs, cached predictions, ablation intermediates, and result exports.
- Large checkpoints and large processed biological/perturbation matrices.
- Nested Git metadata, archive files, macOS metadata, Windows Zone.Identifier streams, and Python caches.

## Author checks before publication

1. Add the chosen open-source license.
2. Confirm redistribution rights or public download locations for processed biological datasets.
3. Add the final paper URL/DOI and update `CITATION.cff`.
4. Run each final experiment config in the publication environment and archive the exact checkpoints separately.
