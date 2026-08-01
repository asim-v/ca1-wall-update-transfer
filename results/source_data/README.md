# Tracked source data

These compact files support the repository's tracked reports. Raw animal
files, large intermediate fits, and regenerable diagnostics are excluded from
Git.

- `plus_cohort_summary.json` / `.csv` — frozen per-sequence and animal-level
  descriptive summaries, including input hashes and provenance roles.
- `plus_common_support_exploratory.json` — post-outcome common-query,
  equal-neuron, and segment-label diagnostics.
- `plus_behavior_nuisance_control.json` — post-outcome joint speed and
  allocentric movement-direction nuisance adjustment (historical JSON field
  names use “heading”).
- `plus_trace_shift_B19_summary.json` — compact summary and hashes for staged
  circular trace-position diagnostics in the two clean prospectively locked
  animals.

Some frozen machine-readable role tokens use the historical word
`preregistered`. In this repository they mean only "prospectively locked in
version control at commit `443d038`." They are not evidence of an independently
timestamped public preregistration and are retained to preserve the frozen
artifact lineage.

Regenerate the first three with:

```powershell
.venv\Scripts\python.exe scripts\summarize_plus_cohort.py --output-prefix results\source_data\plus_cohort_summary

.venv\Scripts\python.exe scripts\run_exploratory_common_support.py data\raw\QLAK-CA1-08.complete.mat data\raw\QLAK-CA1-30.complete.mat data\raw\QLAK-CA1-50.complete.mat data\raw\QLAK-CA1-56.complete.mat --output results\source_data\plus_common_support_exploratory.json --figure results\source_data\plus_common_support_exploratory.png --subset-neurons 47 --subset-draws 200 --seed 20260729

.venv\Scripts\python.exe scripts\run_plus_behavior_nuisance_control.py --animal-files data\raw\QLAK-CA1-08.complete.mat data\raw\QLAK-CA1-30.complete.mat data\raw\QLAK-CA1-50.complete.mat data\raw\QLAK-CA1-56.complete.mat --source-results results\diagnostics\QLAK-CA1-08_plus_boundary.json results\diagnostics\QLAK-CA1-30_plus_boundary.json results\diagnostics\QLAK-CA1-50_plus_boundary.json results\diagnostics\QLAK-CA1-56_plus_boundary.json --output results\source_data\plus_behavior_nuisance_control.json --figure results\source_data\plus_behavior_nuisance_control.png
```

The trace-shift summary points to SHA-256 hashes of the complete ignored
diagnostic outputs, which can be regenerated with
`scripts/run_plus_trace_shift_control.py`.

## Discovery-track artifacts

Several artifact and script filenames retain the early working word
`component` so that provenance links do not break. The audited scientific
claim is an exact-location wall-conditioned cell profile, not a portable
compositional wall component.

- `boundary_fragment_screen.json` — within-exposure, within-global-shape-pair
  contrast between exact seams walled in both shapes and seams that change
  between wall and open.
- `boundary_component_validation.json` — target-rate-held-out local prediction
  across exposure cycles with non-overlapping outer square baselines,
  test-aware target labels/support, exact-location open controls,
  session-global-rate demeaning, and orientation summaries.
- `boundary_fragment_raw_split.json` — contiguous half-session reconstruction
  from raw positions and event traces, four symmetrically averaged crossed
  assignments with non-overlapping samples on the two correlation sides, a
  stricter dwell sensitivity, and released-map reconstruction audits.
- `boundary_fragment_controls.json` — deterministic near/far, no-square,
  occupancy-weighted, and distinct same-signed-normal wall controls.
- `boundary_fragment_behavior_adjusted.json` — raw-event target-rate-held-out
  local prediction after additive adjustment for speed, allocentric movement
  direction, and session time at a familiar-square-calibrated common
  reference.
- `boundary_fragment_aggregation_sensitivity.json` — equal record,
  exposure-pair, target-environment, and exact-oriented-seam weighting of the
  target-rate-held-out result; this is aggregation robustness, not independent
  confirmation.
- `boundary_fragment_open_reversal.json` — symmetric target-aware control for
  target seams that are open: compare profiles trained from open versus wall
  instances at the exact coordinates in the preceding exposure cycle.
- `boundary_fragment_session_permutation.json` — one coherent global-cell-ID
  permutation per animal and draw, shared across dependent records and
  constrained to preserve every record-specific common-cell set; descriptive
  identity diagnostic only.
- `boundary_component_figure.csv` — the exact animal-level values plotted in
  `reports/figures/boundary_component_discovery.png`.
- `alias_resolution_screen.json`, `alias_split_half_screen.json`, and
  `alias_identity_decoder.json` — falsification artifacts for the rejected
  exact-local-boundary-alias branch.
- `field_evacuation_screen.json` — falsification artifact for blocked-field
  relocation; prospective learned-suppression controls are summarized in
  `DISCOVERY.md`.

### Stronger-claim artifacts

- `boundary_fragment_context_matched.json` — target-rate-held-out focal
  wall/open prediction after exactly matching every other neighbor state of
  the accessible target partition.
- `boundary_fragment_single_tile_counterfactual.json`, `_near.json`, and
  `_far.json` — the exact `u` versus `o` one-tile edit at the same held-out
  seam and its spatial decay.
- `boundary_fragment_single_tile_counterfactual_behavior_adjusted.json` —
  raw-event single-tile prediction after additive speed, allocentric
  movement-direction, and session-time adjustment.
- `boundary_fragment_single_tile_order_placebo.json` — generic same-seam
  later-minus-earlier drift and drift-adjusted `u-o` sensitivities.
- `boundary_fragment_cross_location_transfer.json` — prediction at a
  non-overlapping seam one 25 cm grid step away.
- `boundary_fragment_cross_location_spatial_controls.json` and
  `boundary_fragment_cross_location_cell_permutation.json` — strip-overlap,
  layout-matched direction, target-state specificity, and coherent
  registered-cell identity audits.
- `boundary_fragment_cross_location_behavior_adjusted.json` — raw-event
  kinematic adjustment of the translated prediction.
- `boundary_fragment_cross_location_behavior_adjusted_eligibility_audit.json`
  — comparison of stored-map and adjusted target query/session/seam identities
  plus common-cell and target/source-bin counts.
- `boundary_fragment_cross_location_mirror_open.json` — frozen same-session
  reflected open-seam control matched for source distance, translation axis,
  spatial lag, paired bins, and registered cells.
- `boundary_fragment_cross_location_spatial_null.json` - exact animal-level
  sign-flip calibration of the equal-midpoint wrong-orientation and strict
  same-session reflected-open spatial controls. It preserves the observed
  neural maps and treats animals, not repeated queries, as the independent
  units; its tail fractions are descriptive because the study is exploratory.
- `boundary_compositional_counterfactual.json` — design-rank audit and
  leakage-safe two-wall additive test.
- `boundary_fragment_experience_did.json` and
  `boundary_fragment_experience_raw_split.json` — stored-map and raw-half
  experience difference-in-differences against open/open reliability.
- `blair_barrier_scar.json` — independently locked former-barrier scar test.
- `local_wall_update_transfer_figure.csv` — exact values plotted in
  `reports/figures/local_wall_update_transfer.png`.
- `partial_generalization_explainer.csv` — exact animal-level values plotted
  in `reports/figures/partial_generalization_explainer.png`, including the
  exact-versus-cross-location comparison and spatial-control boundaries.

Regenerate the surviving boundary-component results with:

```powershell
.venv\Scripts\python.exe scripts\run_boundary_fragment_screen.py --data-dir data\raw --output results\source_data\boundary_fragment_screen.json

.venv\Scripts\python.exe scripts\run_boundary_component_validation.py --data-dir data\raw --output results\source_data\boundary_component_validation.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_raw_split.py --data-dir data\raw --output results\source_data\boundary_fragment_raw_split.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_controls.py --data-dir data\raw --output results\source_data\boundary_fragment_controls.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_aggregation_sensitivity.py --input results\source_data\boundary_component_validation.json --output results\source_data\boundary_fragment_aggregation_sensitivity.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_open_reversal.py --data-dir data\raw --output results\source_data\boundary_fragment_open_reversal.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_session_permutation.py --data-dir data\raw --output results\source_data\boundary_fragment_session_permutation.json --permutations 999 --seed 20260730

.venv\Scripts\python.exe scripts\run_boundary_fragment_behavior_adjusted.py --data-dir data\raw --output results\source_data\boundary_fragment_behavior_adjusted.json --trace-cell-chunk 64

.venv\Scripts\python.exe scripts\make_boundary_component_figure.py
```

Regenerate the stronger-claim package with:

```powershell
.venv\Scripts\python.exe scripts\run_boundary_fragment_context_matched.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_single_tile_counterfactual.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_single_tile_counterfactual.py --minimum-bins 4 --depths-cm 2.5 7.5 --output results\source_data\boundary_fragment_single_tile_counterfactual_near.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_single_tile_counterfactual.py --minimum-bins 4 --depths-cm 17.5 22.5 --output results\source_data\boundary_fragment_single_tile_counterfactual_far.json

.venv\Scripts\python.exe scripts\run_boundary_fragment_behavior_adjusted.py --output results\source_data\boundary_fragment_single_tile_counterfactual_behavior_adjusted.json --match-nonfocal-context --match-global-counterfactual --animal-cache-dir results\diagnostics\single_tile_behavior_adjusted_checkpoints

.venv\Scripts\python.exe scripts\audit_boundary_fragment_single_tile_order.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_cross_location_transfer.py

.venv\Scripts\python.exe scripts\audit_boundary_fragment_cross_location_spatial_controls.py

.venv\Scripts\python.exe scripts\audit_boundary_fragment_cross_location_cell_permutation.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_cross_location_behavior_adjusted.py

.venv\Scripts\python.exe scripts\audit_boundary_fragment_cross_location_behavior_adjusted_eligibility.py

.venv\Scripts\python.exe scripts\audit_boundary_fragment_cross_location_mirror_open.py

.venv\Scripts\python.exe scripts\summarize_boundary_fragment_cross_location_spatial_null.py

.venv\Scripts\python.exe scripts\audit_boundary_compositional_counterfactual.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_experience_did.py

.venv\Scripts\python.exe scripts\run_boundary_fragment_experience_raw_split.py

.venv\Scripts\python.exe scripts\run_blair_barrier_scar.py

.venv\Scripts\python.exe scripts\make_local_wall_update_figure.py

.venv\Scripts\python.exe scripts\make_partial_generalization_explainer.py
```

The restricted cell-permutation draws probe dependence on registered-cell
alignment while preserving overlapping common-cell sets. Exchangeability is
limited, so the tail fractions are descriptive diagnostics, not proof of
necessity or population inference. They do not turn seams or cells into
biological replicates; animal-level directions are reported separately.
