# Reusable CA1 Population-Update Directions Across Boundary and Reward Changes

This repository contains the boundary-analysis code, derived source data,
audit reports, manuscript, and release artifacts for a linked reanalysis of
longitudinal CA1 activity. The companion reward-relocation analysis is public
at [`asim-v/ca1-goal-update-transfer`](https://github.com/asim-v/ca1-goal-update-transfer).

## Main result

Across two independent datasets, a registered-cell population-change direction
estimated at one absolute location predicts a related change elsewhere.

- **Boundary arm:** a wall-minus-open source vector predicts a non-overlapping
  target seam in 7/7 mice (mean correlation 0.148). Prediction is weaker than
  the matched same-seam benchmark (0.230) and remains positive after behavioral
  adjustment (0.119; 7/7 mice).
- **Reward arm:** two same-signed 120 cm reward moves have more aligned update
  fields than an opposite-signed 240 cm return (8/8 confirmatory mice; exact
  one-sided p = 0.003906). A nonlinear out-of-fold speed and licking model
  retains a smaller dF/F effect in 7/8 mice.

The estimators are reported separately and are not numerically pooled. The
evidence supports structured reuse with spatial and behavioral dependence, not
a universal, coordinate-free, or behavior-independent map operator.

## Manuscript

The active two-column manuscript is:

> *Reusable CA1 population-update directions during boundary and reward
> changes: convergent evidence from two longitudinal datasets*

- [Compiled manuscript](manuscript/ca1_reusable_population_updates_manuscript.pdf)
- [LaTeX source](manuscript/main.tex)
- [Bibliography](manuscript/references.bib)
- [arXiv metadata](submission/ARXIV_METADATA.md)
- [Validated arXiv source package](submission/ca1_reusable_population_updates_arxiv.zip)

## Repository contents

- `reports/local_wall_update_transfer.md` - boundary evidence and limitation
  ledger.
- `reports/figures/` - four manuscript image assets; the common analysis
  schematic is generated directly in LaTeX.
- `results/source_data/` - machine-readable boundary results and audit
  artifacts.
- `scripts/` - boundary analysis and figure-generation entry points.
- `src/ca1_geometry/` - reusable boundary-analysis implementation.
- `tests/` - regression and synthetic tests.
- `submission/` - arXiv metadata and the exact validated source archive.
- `DATA.md` - source provenance and redistribution boundaries.

Reward-specific protocols, scripts, registration audits, trial-level behavior
models, and machine-readable results are maintained in the
[companion repository](https://github.com/asim-v/ca1-goal-update-transfer).

## Reproduce the boundary environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m pytest -q
```

The repository includes derived outputs, but not the large raw imaging files.
Download source recordings from the archives listed in [DATA.md](DATA.md)
before rerunning raw-data stages.

## Claim boundary

- Animals, not cells, bins, queries, or resamples, are the biological units.
- The boundary arm remains exploratory and does not eliminate every smooth
  spatial-continuity explanation.
- The reward arm confounds displacement sign with step length.
- Speed and licking carry strong reward-displacement structure; residual neural
  specificity is not evidence of behavioral independence or causal mediation.
- The datasets do not establish composition, innateness, portable amplitude,
  or a universal hippocampal primitive.

## Author

Javier Emilio Bazan Sanchez  
Facultad de Ciencias, Universidad Nacional Autonoma de Mexico (UNAM)  
[bazan@ciencias.unam.mx](mailto:bazan@ciencias.unam.mx)

## Status

Working scientific release, August 2, 2026. The boundary analysis is
exploratory. The reward endpoint is confirmatory relative to its declared
local development and holdout workflow; neither arm is an independent
prospective replication.
