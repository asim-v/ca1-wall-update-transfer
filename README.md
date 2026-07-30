# Limited spatial transfer of a CA1 wall-related population update

This repository contains the manuscript, analysis code, derived source data,
and audit reports for an exploratory longitudinal reanalysis of dorsal CA1
calcium-imaging data.

## Main finding

A registered-cell wall-minus-open response learned from training geometries
predicts a later wall response:

- At the same physical seam, the context-matched wall-open correlation
  advantage is 0.184 and positive in 7/7 animals.
- A single-tile counterfactual gives an advantage of 0.225, positive in 7/7.
- At a non-overlapping seam 25 cm away, the shifted source effect predicts the
  later target residual at mean correlation 0.148, positive in 7/7.
- The query-matched exact-location benchmark is stronger at 0.230, giving a
  shifted-minus-exact penalty of -0.082.
- Behavior-adjusted shifted prediction remains positive in 7/7 animals at
  mean correlation 0.119.

The supported interpretation is limited spatial transfer of a cell-resolved
CA1 wall-related update. The data do not establish a location-invariant wall
primitive, linear composition, learning, innateness, or a persistent
former-barrier scar.

## Contents

- `manuscript/` - LaTeX manuscript and bibliography.
- `manuscript/ca1_wall_update_transfer_manuscript.pdf` - compiled working
  manuscript.
- `reports/local_wall_update_transfer.md` - full evidence and limitation
  ledger.
- `reports/figures/` - manuscript-ready primary figure.
- `results/source_data/` - tracked derived result tables and JSON audits.
- `scripts/` - analysis and audit entry points used by the active result.
- `src/ca1_geometry/` - reusable analysis implementation.
- `tests/` - regression and synthetic tests.
- `DATA.md` - source-data provenance and redistribution boundary.

## Reproduce the code environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.venv\Scripts\python.exe -m pytest -q
```

The public repository includes derived analysis outputs but not the large raw
calcium-imaging files. Download those from the original archives described in
`DATA.md` before rerunning raw-data scripts.

## Build the manuscript

From `manuscript/`:

```powershell
tectonic -X compile main.tex --outdir ..\output\pdf
```

## Author

Javier Emilio Bazan Sanchez  
Facultad de Ciencias, Universidad Nacional Autonoma de Mexico (UNAM)  
[bazan@ciencias.unam.mx](mailto:bazan@ciencias.unam.mx)

## Status

Working scientific release, July 30, 2026. The analysis is exploratory and
internally held out, not preregistered or independently replicated.
