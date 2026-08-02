# LaTeX manuscript

This directory contains the active working paper:

> *Reusable CA1 population-update directions during boundary and reward
> changes: convergent evidence from two longitudinal datasets*

The build uses a compact two-column layout with five figures and a full-width
evidence table. Boundary and reward estimators remain separate; no numerical
effect is pooled across datasets.

## Contents

- `main.tex` - complete working manuscript.
- `references.bib` - DOI-checked bibliography.
- `ca1_reusable_population_updates_manuscript.pdf` - compiled 12-page paper.
- `../reports/figures/` - the five manuscript figures.
- `../results/source_data/` - boundary figure source data and audits.
- `../reports/local_wall_update_transfer.md` - boundary evidence ledger.
- [Reward companion](https://github.com/asim-v/ca1-goal-update-transfer) -
  reward protocols, code, machine-readable results, and audit trail.

## Build

Compile from this directory so the bibliography and figure paths resolve:

```powershell
tectonic -X compile main.tex --outdir ..\output\pdf
Copy-Item ..\output\pdf\main.pdf `
  ..\output\pdf\ca1_reusable_population_updates_manuscript.pdf
```

With TeX Live:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Reporting rules

- Animals, not cells or queries, are the biological units.
- Cross-location boundary prediction is weaker than exact-location reuse.
- The mirror-open control covers five animals and tangential translation only.
- The reward comparison confounds displacement sign with step length.
- Speed and licking share reward-displacement geometry; residual neural
  specificity does not establish behavioral independence.
- The two dataset-specific estimators cannot be numerically pooled.
