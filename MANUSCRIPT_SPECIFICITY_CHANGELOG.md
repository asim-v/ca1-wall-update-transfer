# Manuscript specificity changelog

Revision date: 2026-08-02

Starting manuscript commit: `fa84fe5` (significance statement removed), with
the direct-publication workflow subsequently recorded at `29f569a`.

Audit commit: `b99d80f`

Frozen positive-model protocol commit: `dac65b4`

Positive-model implementation commits: `bf7d438`, `d992fe6`, and `ef1208a`

Frozen result commit: `cbccce7`

Manuscript revision commit: `565e1b2`

Public release commit: `03eaaf7`

## Central claim

Previous rhetorical center:

> Related boundary and reward changes reveal reusable CA1 population-update
> directions across locations.

Revised central claim:

> CA1 remapping contains a registered-cell component that transfers across
> repeated spatial relations, but it is weaker than exact-location recurrence
> and remains constrained by position and behavioral state.

The revision treats same-location recurrence, partial cross-location transfer,
and an invariant transformation as distinct evidential levels. The middle
level is the empirical center.

## Claims narrowed

- Removed affirmative framing around a reusable update rule.
- Replaced reward-displacement language with transition-class language because
  the available return transition differs in both sign and distance.
- Recast wall and reward results as conceptual convergence rather than a pooled
  numerical replication.
- Made explicit that attenuation of roughly half the reward-specificity
  statistic is not an estimate of variance explained in CA1 activity.
- Stated that stable boundary-relative, reward-relative, or smooth place-field
  tuning may generate the measured update directions.
- Kept wall analyses exploratory and the new empirical baseline
  `frozen-local`; no prospective-confirmatory wording was added.
- Did not reintroduce a significance statement, following the author's
  explicit decision in commit `fa84fe5`.

## Claims strengthened

- Put the non-overlapping cross-location wall prediction first in the abstract
  and discussion.
- Made the exact-location benchmark and shifted-minus-exact decrement central:
  mean shifted correlation `0.148`, exact correlation `0.230`, and
  shifted-minus-exact difference `-0.082`.
- Added the fully enumerated empirical exact-distance baseline. The correct
  relation had mean animal-level percentile rank `0.612`, above chance in
  `7/7` animals, and exceeded the mean alternative in `7/7`.
- Reported the negative boundary of that result beside it: the correct relation
  exceeded the best alternative in only `5/7` animals, and the strict
  tangential tier remained heterogeneous.
- Clarified why registered-cell identity makes the result more specific than
  elevated population firing near a wall or reward.

## Analyses added or adjudicated

- Added the frozen empirical spatial-rank analysis from
  `results/positive_model_spatial_rank_v1.json`.
- Reported all Tier-1 and support-limited Tier-2 outcomes, including raw-rate
  sensitivities and failed companion endpoints, in
  `POSITIVE_MODEL_ADJUDICATION_RESULTS.md`.
- Applied the predeclared stop to the fitted M0--M3 boundary comparison. The M3
  calibration weight could not be learned on a pseudoquery disjoint from both
  outer neural sessions for every animal and outer pair. No partial fitted-model
  target endpoint was inspected.
- Left reward R0--R4 unexecuted; the existing nonlinear behavior analysis was
  not relabeled as the protocol's combined positive model.

## Figures changed

- Added `reports/figures/positive_model_spatial_rank_v1.png`, showing the full
  source enumeration, animal-level percentile ranks, correct-minus-average and
  correct-minus-best alternatives, and the strict tangential tier.
- Added its tracked animal-level source table at
  `results/source_data/positive_model_spatial_rank_v1_figure.csv` and builder at
  `scripts/make_positive_model_spatial_rank_figure.py`.
- Retained paired exact-versus-shifted and mirror-open panels so the transfer
  decrement and support limits remain visually explicit.
- Regenerated older wall figures with ``exact recurrence'' and ``partial
  transfer'' in place of the broader reuse/generalization labels.
- Renamed the reward robustness figure ``reward-transition specificity'' in the
  companion repository and pushed the reproducible figure change at
  `61d76a5`.

## Manuscript structure changed

1. Established boundary- and reward-relative coding as prior knowledge.
2. Defined the narrower longitudinal registered-cell difference question.
3. Used same-seam recurrence as an internal validation.
4. Presented cross-location wall prediction and the exact-location benchmark
   as the empirical spine.
5. Added full empirical spatial enumeration and the fitted-model stop.
6. Presented reward-transition alignment as independent conceptual
   convergence.
7. Placed behavior-only and nonlinear residual results immediately beside the
   reward claim.
8. Moved the single-tile counterfactual and order control to the supplementary
   exploratory section.

## Unresolved limitations

- A fitted boundary-relative/place-field model has not been beaten on a
  leakage-free held-out target endpoint.
- The strict tangential baseline is heterogeneous and has uneven support; one
  animal is ineligible and another contributes only three queries.
- The mirror-open control covers five animals and tangential translations only,
  with limited raw-rate robustness.
- Wall layout order was fixed and cannot identify learning, stabilization, or
  innate structure.
- Reward displacement sign is confounded with displacement distance.
- Measured speed and licking share substantial reward-transition geometry;
  unmeasured task-state variables remain.
- Longitudinal registration selects a stable-cell subset in both reanalyses.

## Quantitative traceability

`results/abstract_quantitative_claim_traceability_v1.json` maps every
quantitative abstract claim to a JSON pointer, transformation, analysis
classification, and source result file. The reward files are pinned to public
repository commit `2d97ee7` with SHA-256 hashes. The table records that no
significance statement is present. It is validated by
`scripts/validate_abstract_claim_traceability.py`.

## Exact manuscript-revision files

- `manuscript/main.tex`
- `manuscript/references.bib`
- `manuscript/README.md`
- `reports/figures/positive_model_spatial_rank_v1.png`
- `reports/figures/local_wall_update_transfer.png`
- `reports/figures/partial_generalization_explainer.png`
- `reports/figures/adversarial_controls_v1.png`
- `results/source_data/positive_model_spatial_rank_v1_figure.csv`
- `scripts/make_local_wall_update_figure.py`
- `scripts/make_partial_generalization_explainer.py`
- `scripts/make_positive_model_spatial_rank_figure.py`
- `results/abstract_quantitative_claim_traceability_v1.json`
- `scripts/validate_abstract_claim_traceability.py`
- `MANUSCRIPT_SPECIFICITY_CHANGELOG.md`
- `README.md`
- `DECISIONS.md`
- `MILESTONES.md`
- `submission/arxiv/main.tex`
- `submission/arxiv/references.bib`
- `submission/arxiv/main.bbl`
- `submission/arxiv/figures/*.png`
- `submission/README.md`
- `submission/ca1_reusable_population_updates_arxiv.zip`
- `output/pdf/ca1_reusable_population_updates_manuscript.pdf`

The active source and exact extracted arXiv ZIP both compile to 14 letter-size
pages. All pages were rendered and inspected after the final figure rebuild.
