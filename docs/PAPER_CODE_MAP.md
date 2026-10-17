# Manuscript-to-code map

| Manuscript component | Implementation |
|---|---|
| Mean velocity and mass-growth networks \(v_\theta,h_\phi\) | `wfr_mfm/models.py` |
| Uniform time-pair mixture, Algorithm 4 | `wfr_mfm/time_sampling.py` |
| WFR OET coupling and mini-batch OET | `wfr_mfm/transport.py` |
| Traveling-Dirac WFR conditional path | `wfr_mfm/paths.py` |
| Derivative-identity regression target | `wfr_mfm/objectives.py` |
| WFR-MFM training, Algorithm 2 | `wfr_mfm/training.py` |
| One-step and multi-step updates, Algorithm 1 | `wfr_mfm/inference.py` |
| Weighted \(W_1\) and RME | `wfr_mfm/metrics.py` |
| Conditional perturbation extension, Algorithm 3 | `wfr_mfm/conditional.py` |

## Experiment mapping

| Manuscript result | Entry point and configuration |
|---|---|
| Tables 1–2, Gene | `scripts/train.py`, `scripts/benchmark.py`, `configs/gene.yaml` |
| Table 2 / Fig. 5, Dyngen | `scripts/train.py`, `configs/dyngen.yaml` |
| Table 2 / Fig. 6, Gaussian 1000D | `scripts/generate_gaussian.py`, `configs/gaussian.yaml` |
| Fig. 1 / Table 8, EMT | `scripts/train.py`, `configs/emt.yaml` |
| Fig. 3 / Tables 9–12, EB | `configs/eb_5d.yaml`, `eb_50d.yaml`, `eb_100d.yaml` |
| Fig. 2 speed–accuracy curve | `scripts/benchmark.py --steps 1 2 5 10 20 30` |
| Table 13, CITE-seq | `configs/cite.yaml`; use `--steps-per-interval 10` for the reported multi-step model |
| Table 14, Mouse | `configs/mouse.yaml` |
| Fig. 4 / Sec. C.12, perturbations | `scripts/generate_perturbation.py`, `train_conditional.py`, `infer_conditional.py` |

Baseline methods in the comparison tables remain external dependencies of the scientific comparison and are not part of WFR-MFM itself. Their copied source trees are therefore excluded from this release.
