# LaTeX manuscript

This directory contains the active working paper:

> *Limited spatial transfer of a reusable CA1 wall-related population update:
> an exploratory longitudinal reanalysis*

The manuscript follows the project's current evidence hierarchy. Its central
claim is limited cross-location reuse of a cell-resolved CA1 wall-related
update. It does not claim a location-invariant primitive, linear composition,
experience-dependent stabilization, or innateness.

## Contents

- `main.tex` - complete working manuscript.
- `references.bib` - DOI-checked bibliography.
- `../reports/figures/local_wall_update_transfer.png` - tracked primary figure.
- `../results/source_data/local_wall_update_transfer_figure.csv` - primary
  figure source data.
- `../reports/local_wall_update_transfer.md` - scientific source of truth and
  complete limitation ledger.

The previous boundary-normal manuscript remains recoverable from Git history
through milestone `milestone-manuscript-2026-07-29`. Funding and the CRediT
statement remain placeholders.

## Build

Compile from this directory so bibliography and figure paths resolve:

```powershell
Set-Location manuscript
tectonic -X compile main.tex --outdir ..\output\pdf
Copy-Item ..\output\pdf\main.pdf `
  ..\output\pdf\ca1_wall_update_transfer_manuscript.pdf
```

With TeX Live:

```powershell
Set-Location manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
New-Item -ItemType Directory -Force ..\output\pdf | Out-Null
Copy-Item main.pdf ..\output\pdf\ca1_wall_update_transfer_manuscript.pdf
```

Stable deliverable:

```text
output/pdf/ca1_wall_update_transfer_manuscript.pdf
```

## Reporting rules

- Animals, not cells or queries, are the biological units.
- The cross-location estimator is separate from the `u-o` single-tile
  estimator.
- The cross-location audit matches query/session/seam identities and support
  counts; it does not hash full cell-ID or bin-coordinate arrays.
- The mirror-open control covers five animals and tangential translation only.
- Design locks recorded in local Git are process-declared, not independently
  timestamped preregistrations.
- The data do not identify experience effects or innateness.
