# LaTeX manuscript

This directory contains the active working paper:

> *Partial transfer of cell-resolved CA1 remapping directions across boundary
> and reward changes*

The current build uses a compact two-column layout with full-width figures and
evidence table.

The manuscript combines two independent longitudinal reanalyses. The boundary
dataset tests whether a cell-resolved wall contrast predicts a new physical
location, benchmarks it against exact-location recurrence, and ranks the
predicted relation among every admissible exact-distance source. The reward
dataset tests repeated transition-class alignment, including frozen drift/null
controls and nonlinear trial-level conditioning on speed and licking. The
estimators remain separate; no cross-dataset effect size is pooled.

## Contents

- `main.tex` - complete working manuscript.
- `references.bib` - DOI-checked bibliography.
- `../reports/figures/local_wall_update_transfer.png` - tracked primary figure.
- `../reports/figures/partial_generalization_explainer.png` - conceptual and
  paired-data explanation of the main result and its spatial limits.
- `../reports/figures/positive_model_spatial_rank_v1.png` - full empirical
  exact-distance source enumeration and its support-limited tangential tier.
- `../reports/figures/adversarial_controls_v1.png` -
  drift, trial/label null, preprocessing, and fixed-condition controls.
- `../reports/figures/trial_behavior_model_v1.png` -
  nonlinear out-of-fold speed and licking analysis.
- `../results/source_data/local_wall_update_transfer_figure.csv` - primary
  figure source data.
- `../results/source_data/partial_generalization_explainer.csv` - explanatory
  figure source data.
- `../results/source_data/positive_model_spatial_rank_v1_figure.csv` -
  animal-level source data for the enumerated spatial-baseline figure.
- `../results/abstract_quantitative_claim_traceability_v1.json` - sentence-level
  mapping from every quantitative abstract claim to its source result.
- `../reports/local_wall_update_transfer.md` - scientific source of truth and
  complete limitation ledger.

The previous boundary-normal manuscript remains recoverable from Git history
through milestone `milestone-manuscript-2026-07-29`.

## Build

Compile from this directory so bibliography and figure paths resolve:

```powershell
Set-Location manuscript
tectonic -X compile main.tex --outdir ..\output\pdf
Copy-Item ..\output\pdf\main.pdf `
  ..\output\pdf\ca1_reusable_population_updates_manuscript.pdf
```

With TeX Live:

```powershell
Set-Location manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
New-Item -ItemType Directory -Force ..\output\pdf | Out-Null
Copy-Item main.pdf ..\output\pdf\ca1_reusable_population_updates_manuscript.pdf
```

Stable deliverable:

```text
output/pdf/ca1_reusable_population_updates_manuscript.pdf
```

## Reporting rules

- Animals, not cells or queries, are the biological units.
- The cross-location estimator is separate from the `u-o` single-tile
  estimator.
- The cross-location audit matches query/session/seam identities and support
  counts; it does not hash full cell-ID or bin-coordinate arrays.
- The mirror-open control covers five animals and tangential translation only.
- The fully distance-matched tangential wrong-orientation comparison is
  heterogeneous (3/6 positive); pooled spatial-null evidence must not be
  described as eliminating smooth spatial continuity.
- Design locks recorded in local Git are process-declared, not independently
  timestamped preregistrations.
- The data do not identify experience effects or innateness.
- The reward comparison confounds displacement sign with step length and does
  not establish portable update amplitude.
- Speed and licking carry strong reward-displacement specificity. The primary
  nonlinear residual retains about half the neural mean; this is not evidence
  for behavior independence or causal mediation.
- Wall and reward endpoints use different statistics and cannot be numerically
  pooled.
