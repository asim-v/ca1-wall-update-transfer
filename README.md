# Partial Transfer of Cell-Resolved CA1 Remapping Directions

This repository contains the boundary-analysis code, derived source data,
audit reports, manuscript, and release artifacts for a linked reanalysis of
longitudinal CA1 activity. The companion reward-relocation analysis is public
at [`asim-v/ca1-goal-update-transfer`](https://github.com/asim-v/ca1-goal-update-transfer).

## Main result

Across two independent datasets, a registered-cell remapping direction
estimated at one absolute location partially predicts a related change
elsewhere.

- **Boundary arm:** a wall-minus-open source vector predicts a non-overlapping
  target seam in 7/7 mice (mean correlation 0.148). Prediction is weaker than
  the matched same-seam benchmark (0.230) and remains positive after behavioral
  adjustment (0.119; 7/7 mice). In a frozen empirical spatial baseline, the
  correct relation ranks above chance in 7/7 mice (mean percentile 0.612), but
  it is not reliably better than the best alternative and the strict
  tangential subset is heterogeneous.
- **Reward arm:** repeated 120 cm reward-transition classes have more aligned
  update fields than comparisons with the 240 cm return (8/8 confirmatory
  mice; exact one-sided p = 0.003906). A nonlinear out-of-fold speed and
  licking model retains a smaller dF/F relation in 7/8 mice.

The estimators are reported separately and are not numerically pooled. The
evidence supports partial transfer with spatial and behavioral dependence,
not a universal, coordinate-free, or behavior-independent transformation.

## Manuscript

The active two-column manuscript is:

> *Partial transfer of cell-resolved CA1 remapping directions across boundary
> and reward changes*

- [Compiled manuscript](manuscript/ca1_reusable_population_updates_manuscript.pdf)
- [LaTeX source](manuscript/main.tex)
- [Bibliography](manuscript/references.bib)
- [arXiv metadata](submission/ARXIV_METADATA.md)
- [Validated arXiv source package](submission/ca1_reusable_population_updates_arxiv.zip)

## Repository contents

- `reports/local_wall_update_transfer.md` - boundary evidence and limitation
  ledger.
- `CA1_SPECIFICITY_AND_POSITIONING_AUDIT.md` - claim ladder, novelty matrix,
  reviewer attacks, and wording audit.
- `POSITIVE_MODEL_ADJUDICATION_PROTOCOL.md` and
  `POSITIVE_MODEL_ADJUDICATION_RESULTS.md` - frozen design and complete
  positive-model outcome record.
- `MANUSCRIPT_SPECIFICITY_CHANGELOG.md` - claim-level revision ledger.
- `reports/figures/` - five manuscript image assets; the common analysis
  schematic is generated directly in LaTeX.
- `results/source_data/` - machine-readable boundary results and audit
  artifacts.
- `results/abstract_quantitative_claim_traceability_v1.json` - sentence-level
  provenance for every quantitative claim in the abstract.
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
- The fitted boundary/place-field adjudication could not be scored without
  violating its frozen leakage rule; the empirical spatial baseline is a
  positive control, not a mechanistic separation.
- The reward arm confounds displacement sign with step length.
- Speed and licking carry strong reward-displacement structure; residual neural
  specificity is not evidence of behavioral independence or causal mediation.
- The datasets do not establish composition, innateness, portable amplitude,
  a direction-invariant rule, or a universal hippocampal primitive.

## Author

Javier Emilio Bazan Sanchez  
Facultad de Ciencias, Universidad Nacional Autonoma de Mexico (UNAM)  
[bazan@ciencias.unam.mx](mailto:bazan@ciencias.unam.mx)

## Status

Specificity revision, August 2, 2026. The boundary analysis is exploratory.
The reward endpoint is confirmatory relative to its declared local development
and holdout workflow; neither arm is an independent prospective replication.
All 20 sentence-level quantitative claims in the abstract resolve to pinned
machine-readable result files.
