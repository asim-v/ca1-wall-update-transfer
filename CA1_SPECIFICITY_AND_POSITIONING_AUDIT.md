# CA1 specificity and positioning audit

Audit date: 2026-08-02

Starting repository HEAD: `29f569a472b4512e7843363148a925192e7d3f12`

Starting branch: `codex/transition-error-discovery`

Starting `git status --porcelain`: empty

## Executive verdict

The manuscript contains a real descriptive result: a registered-cell component
of CA1 remapping predicts a related change at a different absolute location.
The result is graded rather than invariant. In the wall dataset, shifted
prediction is positive in every animal but weaker than query-matched recurrence
at the same seam. In the reward dataset, repeated transition classes have
preferentially aligned update fields, but much of the chosen specificity
statistic is shared with speed and licking. Existing analyses establish
**descriptive reuse (interpretation A)**. They do not yet show that empirical
source updates add predictive information beyond a fitted boundary-relative or
reward-relative positive model (interpretation B), and they do not support a
portable abstract operation (interpretation C).

Recommended central claim:

> CA1 remapping contains a registered-cell component that transfers across
> repeated spatial relations, but it is weaker than exact-location recurrence
> and remains constrained by position and behavioral state.

Recommended title:

> **Partial transfer of cell-resolved CA1 remapping directions across boundary
> and reward changes**

This title describes the measured phenomenon without implying a
coordinate-free operation. Of the alternatives in the editorial brief, it is
also the least likely to make the distinct wall and reward estimands sound like
one pooled numerical replication.

## Phase 0: repository and provenance

### Active scientific artifacts

| Role | Tracked path | Status at audit start |
|---|---|---|
| Active manuscript | `manuscript/main.tex` | clean |
| Active bibliography | `manuscript/references.bib` | clean |
| arXiv source mirror | `submission/arxiv/main.tex` | clean |
| arXiv bibliography | `submission/arxiv/references.bib`, `submission/arxiv/main.bbl` | clean |
| Current PDF build target | `output/pdf/ca1_reusable_population_updates_manuscript.pdf` | clean, 1,270,061 bytes |
| Current arXiv source bundle | `submission/ca1_reusable_population_updates_arxiv.zip` | clean, 1,141,803 bytes |
| Boundary public-release checkout | `public-release/ca1-wall-update-transfer` | separate Git checkout, `main` at `4606501` during audit |
| Reward analysis checkout | `project2-goal-update-transfer` | separate Git checkout, clean `main` at `2d97ee7262d540595019217d98c26ae8701a64e2` |

No modified or untracked scientific files were present in either the root
repository or the reward repository at audit start. Existing commits and
branches were not rewritten.

### Figures in the active manuscript and their sources

| Manuscript figure | Rendered asset | Figure builder / immediate table | Upstream machine-readable results |
|---|---|---|---|
| Common analysis logic | TikZ embedded in `manuscript/main.tex` | manuscript source | conceptual only |
| Wall result | `reports/figures/local_wall_update_transfer.png` | `scripts/make_local_wall_update_figure.py`; `results/source_data/local_wall_update_transfer_figure.csv` | context-matched, single-tile, near/far, cross-location, behavior-adjusted, spatial-control, and mirror-open JSON files listed below |
| Partial-transfer explainer | `reports/figures/partial_generalization_explainer.png` | `scripts/make_partial_generalization_explainer.py`; `results/source_data/partial_generalization_explainer.csv` | `boundary_fragment_cross_location_transfer.json`, `boundary_fragment_cross_location_mirror_open.json` |
| Reward adversarial controls | `project2-goal-update-transfer/figures/adversarial_controls_v1.png` | `project2-goal-update-transfer/scripts/plot_adversarial_controls.py` | `project2-goal-update-transfer/results/adversarial_controls_v1.json` |
| Trial-level behavior model | `project2-goal-update-transfer/figures/trial_behavior_model_v1.png` | `project2-goal-update-transfer/scripts/plot_trial_behavior_model.py` | `project2-goal-update-transfer/results/trial_behavior_model_v1.json` |

The wall figure builder reads these tracked result artifacts directly:

- `results/source_data/boundary_fragment_context_matched.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_behavior_adjusted.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_near.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_far.json`
- `results/source_data/boundary_fragment_cross_location_transfer.json`
- `results/source_data/boundary_fragment_cross_location_behavior_adjusted.json`
- `results/source_data/boundary_fragment_cross_location_spatial_controls.json`
- `results/source_data/boundary_fragment_cross_location_mirror_open.json`

Additional controls cited in prose come from:

- `results/source_data/boundary_fragment_cross_location_spatial_null.json`
- `results/source_data/boundary_fragment_cross_location_cell_permutation.json`
- `results/source_data/boundary_fragment_session_permutation.json`
- `results/source_data/boundary_compositional_counterfactual.json`
- `results/source_data/boundary_fragment_experience_did.json`
- `results/source_data/blair_barrier_scar.json`

### Design-lock and result provenance

The boundary analyses are explicitly exploratory. Their code and result files
were first recorded together in `68f6b1b` (cross-location family) or `158fa7a`
(exact-location/session-permutation family); they were not prospectively
registered. The strict spatial-null summary was added in `fe20d6e`. Those
commits are reproducibility landmarks, not confirmatory registrations.

| Endpoint | Design/protocol landmark | Result landmark | Audit classification |
|---|---|---|---|
| Same-seam wall contrast | `158fa7a` | `158fa7a` and `68f6b1b` | exploratory |
| Cross-location wall prediction | `68f6b1b` | `68f6b1b` | exploratory |
| Wall behavior adjustment | `68f6b1b` | `68f6b1b` | exploratory |
| Orientation, mirror-open, cell-permutation controls | `68f6b1b` | `68f6b1b` | exploratory controls |
| Strict tangential spatial null | `fe20d6e` | `fe20d6e` | exploratory sensitivity |
| Reward confirmatory scoring | `b805c56` | `9ddbc03` | frozen-local confirmatory after development mouse |
| Reward registration eligibility | `862e718` | registration ledgers preceding `9ddbc03` | frozen before endpoint scoring |
| Reward adversarial battery | `c4748af` | `7e494ae` (with corrected results carried in `2d97ee7`) | frozen-local control battery |
| Nonlinear trial behavior model | `cce3287` | `2d97ee7` | frozen-local control |

The reward protocol/result separation is inspectable: code and decision gates
were committed before their result commits. This audit does not relabel that
local freezing as a preregistered external study.

## Quantitative traceability audit

All active-manuscript numerical statements were checked against tracked JSON
artifacts. Values below are unrounded source values; manuscript values are the
corresponding stated roundings. Animal summaries, not cells, sessions, seams,
queries, bins, or permutations, are the biological inferential units.

### Boundary results

| Manuscript quantity | Exact source value | Animals/support | Machine-readable source | Audit |
|---|---:|---:|---|---|
| Context-matched same-seam wall-minus-open advantage | 0.1834962013 | 7/7 positive; 252 predictions | `boundary_fragment_context_matched.json` | matches 0.184 |
| Single-tile wall correlation | 0.1969209716 | 7 mice; 46 predictions | `boundary_fragment_single_tile_counterfactual.json` | matches 0.197 |
| Single-tile open correlation | -0.0277374662 | same | same | matches -0.028 |
| Single-tile advantage | 0.2246584377 | 7/7 positive | same | matches 0.225 |
| Raw single-tile advantage | 0.2426969338 | 7/7 positive | same | matches 0.243 |
| Near-strip prediction | 0.2326525331 | 7/7 positive | `boundary_fragment_single_tile_counterfactual_near.json` | matches 0.233 |
| Far-strip prediction | 0.0685010765 | 6/7 positive | `boundary_fragment_single_tile_counterfactual_far.json` | matches 0.069 |
| Cross-location source-to-target prediction, primary demeaned estimator | 0.1476640650 | 7/7 positive; one-grid-step same-normal queries | `boundary_fragment_cross_location_transfer.json` | matches 0.148 |
| Cross-location prediction, raw estimator | 0.1321263701 | 7/7 positive | same | not 0.152; see note below |
| Shifted wall-minus-same-source-open contrast | 0.1524000404 | 7/7 positive | same | matches separately reported 0.152 |
| Shifted source specificity over target-open template | 0.2163963692 | 7/7 positive | `boundary_fragment_cross_location_spatial_controls.json` | matches 0.216 |
| Query-matched exact-location prediction | approximately 0.2297501167, reconstructed per animal as shifted minus (shifted-minus-exact) | 7 mice | `boundary_fragment_cross_location_transfer.json`; figure-source CSV | matches 0.230 |
| Shifted-minus-exact decrement | -0.0820860516 | shifted exceeded exact in 1/7 | same | manuscript reports magnitude 0.082 and direction in prose |
| Behavior-adjusted cross-location prediction | 0.1192391830 | 7/7 positive | `boundary_fragment_cross_location_behavior_adjusted.json` | matches 0.119 |
| Behavior-adjusted specificity over target-open | 0.1718204388 | 7/7 positive | same | matches 0.172 |
| Behavior-adjusted wall-minus-open predictor correlation | 0.1375194460 | 7/7 positive | same | matches 0.138 |
| Correct-minus-incorrect orientation advantage | 0.1009534810 | 7/7 positive; exact sign tail 1/128 | `boundary_fragment_cross_location_spatial_controls.json`, `...spatial_null.json` | matches 0.101 |
| Strict demeaned tangential statistic | 0.0402400179 | 3/6 positive; one-sided exact tail 0.25 | `boundary_fragment_cross_location_spatial_null.json` | matches 0.040 |
| Mirror-open correlation advantage | 0.1137973878 | 5/5 positive; 72 eligible triplets | `boundary_fragment_cross_location_mirror_open.json` | matches 0.114 |
| Absolute source-to-wall mirror subset | 0.0676591765 | 4/5 positive | same | matches 0.068 |
| Within-session wall-minus-mirror-open score | 0.0770670187 | 4/5 positive | same | matches 0.077 |
| Cell-label permutation primary demeaned tails | 0.007, 0.001, 0.001, 0.069, 0.005, 0.002, 0.060 | 7 animals | `boundary_fragment_cross_location_cell_permutation.json` | supports “below 0.01 in five; 0.069 and 0.060 in two” |
| Barrier-scar Pearson center-minus-control | -0.1827547629 | 1/6 positive | `blair_barrier_scar.json` | matches -0.183 |
| Barrier-scar Spearman center-minus-control | 0.0011307559 | 3/6 positive | same | matches 0.001 |

Important estimand distinction: the manuscript's primary 0.148 is the
global-rate-demeaned source-update correlation. The raw version of that same
estimand is 0.132. The nearby value 0.152 is a different contrast, shifted wall
prediction minus a same-source open predictor. The active text currently uses
these consistently, but future summaries must not call 0.152 the “raw 0.148.”

The composition (0.153 versus 0.178; advantage -0.024; 1/5 positive) and
experience (0.045; 4/4 in one stored-map specification) values are traceable to
their tracked exploratory JSON files. They are secondary, support-limited
results and should not be used to expand the central claim.

### Reward results

| Manuscript quantity | Exact source value | Animals/support | Machine-readable source | Audit |
|---|---:|---:|---|---|
| Primary transition specificity, dF/F | 0.0481194655 | 8/8 positive | `project2-goal-update-transfer/results/confirmatory_specificity_v1.json` | matches 0.0481 |
| Mouse-bootstrap 95% interval | [0.0234421869, 0.0757336714] | 8 mice | same | matches [0.0234, 0.0757] |
| Exact animal sign-flip upper-tail P | 0.00390625 | 8 mice | same | matches 0.003906 |
| Event-trace specificity | 0.0400697746 | 8/8 positive | same | matches 0.0401 |
| Pre-switch pseudo-update | 0.0023463725 | 4/8 positive; P=0.359375 | `adversarial_controls_v1.json` | matches 0.00235, P=0.359 |
| Post-switch drift | -0.0011434319 | 3/8 positive; P=0.578125 | same | matches -0.00114, P=0.578 |
| Trial-label null | mean 0.0001000192; 95% interval [-0.0116054, 0.0130781]; 0/199 draws at or above observed; plus-one P=0.005 | 199 draws, animal aggregate retained | same | matches |
| Transition-label null | mean -0.0000064237; 95% interval [-0.0257036, 0.0277525]; P=0.0000549997 | 200,000 draws | same | matches |
| Corrected behavior-only speed specificity | 0.2260663132 | 8/8 positive | `trial_behavior_model_v1.json` | matches 0.2261 |
| Corrected behavior-only lick specificity | 0.3246483380 | 8/8 positive | same | matches 0.3246 |
| Corrected aggregate neural residual | 0.0253230857 | 5/8 positive; P=0.0546875 | same | matches 0.0253, P=0.0547 |
| Frozen nonlinear dF/F residual, lambda=1 | 0.0238345196 | 7/8 positive; P=0.01171875 | same | matches 0.0238, P=0.0117 |
| Frozen nonlinear event residual, lambda=1 | 0.0152991248 | 7/8 positive; P=0.01953125 | same | matches 0.0153, P=0.0195 |
| Held-out MSE improvement, lambda=1 | dF/F 0.0103014292; events 0.0065278309 | 48 sessions | same | matches 1.03% and 0.65% |
| Held-out MSE improvement, lambda=10 sensitivity | dF/F 0.0277393426; events 0.0256023988 | 48 sessions | same | matches 2.77% and 2.56% |
| Registration cohort | 8 eligible of 10 audited; min/median/max common cells 291/523/1142 | 60 sessions | `registration_confirmatory_summary.json` | matches methods ledger |

One provenance nuance should be stated more clearly in the manuscript: the
speed-only value 0.2261 comes from the corrected old-reward-frame result in
`trial_behavior_model_v1.json`. The earlier aggregate value in
`adversarial_controls_v1.json` is 0.2051. The lick value is unchanged. The
active manuscript cites the corrected result, so this is not a numerical
error; it is a version distinction that the claim-source table must preserve.

### Discrepancy disposition

No rounded numerical contradiction was found in the active manuscript. Two
high-risk look-alike values were identified and recorded above:

1. raw cross-location source correlation 0.132 versus the different 0.152
   wall-minus-open contrast;
2. superseded aggregate speed specificity 0.2051 versus corrected 0.2261.

No result file was modified during this audit. Numerical changes, if later
needed for prose clarity, must retain the exact estimand and result-version
mapping.

## Claim ladder

### Level 1 — same-location validation

- **Estimand:** within animal, the correlation advantage of a training-derived
  wall-conditioned local registered-cell vector over the matched open vector
  when predicting a later response at the same oriented seam.
- **Holdout:** target neural rates are excluded from template fitting; training
  and target exposures are non-overlapping. Geometry labels and occupancy are
  used for eligibility.
- **Support:** seven mice; 252 context-matched local predictions in the broad
  analysis; 46 predictions in the single-tile counterfactual.
- **Result:** context-matched mean advantage 0.1835, positive in 7/7; the
  single-tile analysis gives 0.2247, positive in 7/7.
- **Strongest alternative:** stable boundary-relative tuning plus the fixed
  layout sequence, rather than a reusable remapping computation.
- **Strongest control:** exact matching of every other local grid-neighbor
  state; single-tile `u` versus `o` counterfactual; behavior adjustment.
- **Unresolved:** wall presence remains coupled to lost accessible area and
  environment order was not randomized.
- **Justified wording:** “A registered-cell wall contrast recurred at the same
  physical seam across global layouts.”
- **Overclaim:** “CA1 learned a portable wall operation.”

### Level 2 — primary cross-location wall result

- **Estimand:** animal-mean Spearman correlation between a training-derived
  wall-minus-open registered-cell direction at a source seam and the held-out
  target-wall residual at a distinct seam exactly 25 cm away with the same
  signed normal.
- **Holdout:** target test neural rates enter only the evaluation vector;
  source templates use separate training exposures and baselines. Target
  labels/occupancy/registration determine eligibility.
- **Support:** seven mice; source and target strips are non-overlapping; target
  queries and source seams are reused estimator internals, not replicates.
- **Result:** mean 0.1477, positive in 7/7; behavior-adjusted mean 0.1192,
  positive in 7/7.
- **Strongest alternative:** stable boundary-relative fields and generic
  spatial smoothness produce correlated changes at neighboring seams.
- **Strongest control:** same cells/support across alternatives; non-overlap;
  wall-minus-open source contrast; orientation controls; same-session
  mirror-open comparison where available.
- **Unresolved:** no fitted positive boundary/place model has yet been compared
  with the empirical source vector on the same held-out targets.
- **Justified wording:** “A wall-minus-open pattern estimated at one seam
  predicted a later response of the same registered cells at another seam.”
- **Overclaim:** “The wall-update vector transports independently of place.”

### Level 3 — defining spatial limit

- **Estimand:** within-query, within-animal difference between the shifted
  source prediction and an exact-target-location predictor constructed on
  identical cell and spatial support.
- **Holdout:** both predictors are evaluated against the same target neural
  outcome; templates are training-derived.
- **Support:** seven mice.
- **Result:** shifted-minus-exact mean -0.0821; shifted exceeds exact in only
  1/7 mice. Reconstructed exact mean is approximately 0.230.
- **Strongest alternative:** the decrement is partly a reliability or source-
  support difference rather than intrinsic place dependence.
- **Strongest control:** query matching and forced identical cell/spatial
  support for the competing predictors.
- **Unresolved:** no held-out noise ceiling or stable per-animal transfer
  fraction has been frozen; the strict tangential subset is heterogeneous
  (3/6, mean 0.0402, exact one-sided tail 0.25).
- **Justified wording:** “Cross-location prediction was consistently weaker
  than query-matched recurrence at the exact location.”
- **Overclaim:** “The translated component has a fixed portable amplitude.”

### Level 4 — convergent reward result

- **Estimand:** for each mouse and visual environment, correlation of the two
  same-signed 120 cm registered-cell update fields minus the mean correlation
  of each with the opposite-signed 240 cm return update; environments are then
  averaged within mouse and mice within cohort.
- **Holdout:** the endpoint was fixed after a separate development mouse;
  registered-cell eligibility was locked before confirmatory scoring. The
  specificity statistic itself compares observed transitions rather than
  predicting an unseen fourth transition.
- **Support:** eight confirmatory mice, two environments per mouse, 14/16
  environment sequences positive; animals are the inferential units.
- **Result:** mean S=0.04812, bootstrap 95% interval [0.02344, 0.07573], 8/8
  positive, exact sign-flip P=0.003906; events S=0.04007, 8/8.
- **Strongest alternative:** a mixture of track-relative, reward-relative,
  reliability, displacement-distance, and behavioral structure produces the
  alignment.
- **Strongest control:** event traces; registered drift and fixed-condition
  tests; trial- and transition-label nulls; preprocessing/registration
  sensitivity; cell subsampling; leave-one-mouse-out analyses.
- **Unresolved:** signed direction is confounded with distance (120 versus 240
  cm); no independent fourth transition identifies sign and magnitude.
- **Justified wording:** “Repeated reward-transition classes produced
  preferentially aligned registered-cell update fields at different absolute
  positions.”
- **Overclaim:** “CA1 implements a signed-displacement operator.”

### Level 5 — behavioral limit

- **Estimand:** the same reward specificity statistic recomputed from neural
  trial residuals after out-of-fold nonlinear prediction from speed and licking
  in the corrected old-reward frame; held-out trial MSE quantifies total
  predictive performance.
- **Holdout:** kernel-ridge fitting is trial-group cross-fitted; lambda=1 is the
  frozen primary regularization; target residuals are out of fold.
- **Support:** eight mice, 48 scored sessions.
- **Result:** dF/F residual S=0.02383, 7/8 positive, P=0.01172; event residual
  S=0.01530, 7/8, P=0.01953. This retains about 49.5% of the original dF/F
  specificity, while held-out total-trial MSE improves only 1.03% for dF/F and
  0.65% for events at lambda=1.
- **Strongest alternative:** omitted behavioral variables (reward consumption,
  acceleration, pupil, posture, engagement) account for the residual geometry.
- **Strongest control:** nonlinear out-of-fold prediction, event traces, and
  regularization sensitivities.
- **Unresolved:** measured behavior is incomplete and behavior/neural dynamics
  may be jointly caused rather than one explaining the other.
- **Justified wording:** “Speed and licking shared substantial transition-class
  geometry with CA1; a smaller registered-cell relation remained after
  out-of-fold conditioning.”
- **Overclaim:** “Behavior explains half of CA1 activity,” or “the remaining
  signal is behavior independent.”

## Interpretation ladder

| Interpretation | Definition | Current status |
|---|---|---|
| A. Descriptive reuse | Related changes have correlated registered-cell directions | **Supported** by both datasets, with the wall result providing the clearest cross-location prediction |
| B. Model-level reuse | Empirical source updates add held-out information beyond known boundary-/reward-relative and smooth-place models | **Not yet established.** Orientation, mirror-open, behavioral, and permutation controls restrict simple alternatives but no fitted positive model currently makes the decisive nested prediction comparison |
| C. Abstract operation | A transformation applies independently of position, behavior, orientation, and amplitude | **Not supported and contradicted in part** by the exact-versus-shifted decrement, heterogeneous tangential control, behavioral attenuation, failed two-wall composition, and sign/distance confound |

## Focused primary-literature positioning

Literature search was refreshed on 2026-08-02. “Adds” below means the narrow
comparison made by this reanalysis, not a priority claim.

| Prior study | What it established | Data level | Static representation or update | What the present work adds | Remaining overlap |
|---|---|---|---|---|---|
| [O'Keefe & Burgess 1996](https://doi.org/10.1038/381425a0) | Same-cell place fields deform predictably with enclosure geometry; wall-distance inputs explain many changes | Rat CA1 single units across boxes | Field change across geometry | Tests a cell-population difference learned at one internal seam against another seam | Stable wall-relative tuning can itself generate the observed update |
| [Hartley et al. 2000](https://doi.org/10.1002/1098-1063(2000)10:4%3C369::AID-HIPO3%3E3.0.CO;2-0) | Boundary-vector inputs quantitatively predict place fields and barrier-induced fields | Computational model fit to place-cell data | Positive field model | Motivates the required M1 positive baseline and asks whether an empirical source update adds held-out information | Current manuscript has not yet beaten this model family |
| [Lever et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2736390/) | Subicular cells encode allocentric distance and direction to boundaries; barriers can duplicate fields | Rat subiculum single units | Static/reference-frame tuning plus perturbation | Uses CA1 registered-cell identity and source-to-target update prediction | Boundary-vector inheritance remains a direct mechanistic account |
| [Wang et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7259364/) | CA1 field edges align with local texture boundaries | Rat CA1 single units | Static local-boundary organization | Tests local wall/open change vectors rather than field-edge prevalence | Both emphasize local boundary control |
| [Muessig et al. 2024](https://www.nature.com/articles/s41467-024-45098-1) | Boundary geometry reshapes subicular BVC tuning; barriers induce excitation and opposite-side inhibition | Adult/developing rat subiculum | Perturbation response | Shows a graded CA1 population relation across distinct internal seams | Their barrier responses make a simple excitatory BVC baseline insufficient but still a strong alternative |
| [Lee et al. 2025](https://doi.org/10.1016/j.neuron.2024.10.027) | In the source wall dataset, remapping RSMs generalize across animals and local boundary distance/direction models predict CA1 representational structure | Longitudinal mouse CA1 calcium imaging, 5,413 cells | Mostly representation/RSM across geometries | Reindexes the same registered cells as source and target **differences**, asks cross-location prediction, and benchmarks shifted against exact recurrence | This is the closest overlap and makes positive-model adjudication mandatory |
| [Gauthier & Tank 2018](https://pubmed.ncbi.nlm.nih.gov/30008297/) | A small CA1/subiculum population consistently codes reward across environments | Mouse CA1/subiculum population imaging | Static reward-relative cell identity/tuning | Tests alignment of full registered-cell update fields across reward moves | Stable reward cells can contribute to update alignment |
| [Sosa et al. 2025](https://www.nature.com/articles/s41593-025-01985-4) | The source reward dataset contains flexible reward-relative sequences that recruit cells over learning while track coding persists | Longitudinal mouse CA1 two-photon imaging | Reference-frame tuning and remapping | Compares the direction of two separate pre/post updates at different absolute positions | Reward-relative fields are the main positive-model alternative; same dataset means no independent replication of the source phenomenon |
| [Dong et al. 2025](https://doi.org/10.1038/s41593-025-01930-5) | CA1 place fields can be space- or goal-referenced after reward shifts, with intermediate shifts and experience-dependent mechanisms | Mouse CA1 place cells | Reward-shift remapping | Scores cell-by-position update alignment and behavioral sharing rather than cell-category proportions | Reinforces mixture/reference-frame interpretation |
| [Yaghoubi et al. 2026](https://www.nature.com/articles/s41586-025-09958-0) | Longitudinal hippocampal reward representations evolve with learning and are captured in part by a temporal-difference place-cell model | Mouse hippocampal longitudinal recordings/model | Learning-dependent reward representation | Current reward result asks recurrence of transition-class update directions | A predictive reward model is another positive baseline; no priority language is safe |
| [Bernardi et al. 2020](https://doi.org/10.1016/j.cell.2020.09.031) | Approximately parallel coding directions in hippocampus/PFC support cross-condition generalization of abstract task variables | Monkey single-unit populations | Representational difference directions | Applies a related geometric question to longitudinal remapping with fixed cell identity and explicit spatial translation | Shared directions are already established conceptually; the present novelty must be dataset/design-specific |
| [Courellis et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11338822/) | Abstract hippocampal population geometry emerges during human inference | Human single units | Learned relational representation | Tests perturbation-linked changes in rodent CA1 rather than static/inference-state geometry | Broad claims about hippocampal abstraction are not novel here |
| [Gonzalez et al. 2019](https://pubmed.ncbi.nlm.nih.gov/30393037/) and [Sheintuch et al. 2017](https://pubmed.ncbi.nlm.nih.gov/29069591/) | Registered-cell imaging reveals coherent map transformations and enables longitudinal identity-level tests | Mouse CA1 calcium imaging / registration method | Longitudinal static maps and map transformations | Uses registered identities to compare source and target **changes**, not only map stability | Cell registration and coherent map change are established tools/phenomena |

The literature supports this exact introductory distinction:

> Previous work showed that firing fields and population states can be
> organized relative to boundaries, rewards, contexts, and task variables.
> The narrower question here is whether the cell-resolved direction of change
> estimated at one absolute location predicts a related change at another.

The matrix does **not** justify “first demonstration.” Lee et al. is especially
close, Bernardi et al. already established parallel hippocampal coding
directions in another task, and the source-data papers already establish the
relevant boundary and reward reference frames.

## Reviewer-attack matrix

| Attack | What survives | What does not | Required response |
|---|---|---|---|
| “This is algebra from stable boundary-relative tuning.” | Descriptive cross-location prediction and exact-versus-shifted decrement | Any claim of an additional remapping primitive | Fit a leakage-free positive boundary/place model and test M1→M3 held-out improvement; otherwise state model-level adjudication is unresolved |
| “Nearby place fields are smooth.” | Non-overlap, same-support orientation controls, correct relation advantage | General continuity rejection; strict tangential subset is 3/6 | Enumerate all admissible matched spatial alternatives with a frozen matching hierarchy; show ranks and support |
| “Fixed order creates a temporal template.” | Session-order placebos and later evaluation constrain a pure order account | Causal wall attribution | Keep fixed-order limitation prominent; only a counterbalanced experiment resolves it |
| “Wall means inaccessible tile.” | Local matching and same-session mirror-open evidence | Separation of visual boundary from access/topology | State the confound; propose independently controlled transparent/impassable versus opaque/passable boundaries |
| “Reward reuse is behavior reuse.” | Nonlinear residual remains positive in 7/8; event robustness | Behavior independence | Lead with behavior-shared geometry and modest held-out variance; do not call residual uniquely neural |
| “Reward sign is actually distance.” | Repeated transition-class specificity | Signed-displacement operator | Say transition class; require equal-distance opposite-sign relocations |
| “Many observations inflate N.” | Animal-level summaries and exact sign flips | Any seam/cell/session-level population inference | Preserve animal as unit everywhere, label within-animal counts as support |
| “Wall and reward are not replications.” | Conceptual convergence on partial registered-cell reuse | Pooled effect or shared numerical statistic | Keep arms separate and explicitly non-equivalent |
| “The paper searched many analyses.” | Frozen reward endpoint/control commits; all wall negatives retained | Confirmatory wall language | Label wall work exploratory and show complete decision history |

## Wording audit

### Strengthen because directly supported

- State cross-location wall prediction first: 0.148 and 7/7 animals.
- State the exact-location benchmark immediately afterward: approximately
  0.230 and a within-animal decrement of 0.082.
- State that reward transition specificity is 8/8 with a locally frozen
  confirmatory endpoint.
- State that nonlinear behavioral conditioning leaves a smaller positive
  relation in 7/8, while explaining only modest held-out trial variance.

### Narrow or replace

| Current/risky expression | Preferred expression | Reason |
|---|---|---|
| reusable population-update directions | partially transferable registered-cell remapping directions | “Reusable” alone hides the exact-versus-shifted loss |
| same signed reward move | repeated reward-transition class | sign is confounded with distance |
| behavior accounted for about half | residual retained about 49.5% of the chosen specificity statistic | avoids implying half of CA1 activity or variance |
| generalizes across locations | predicts a related response across two non-overlapping locations | “Generalizes” can imply broader spatial invariance |
| wall-specific | wall-favoring under the eligible mirror-open comparison | control covers five animals and is not raw-rate robust in all |
| operator / primitive / algebra | cell-resolved update pattern / registered-cell remapping direction | mechanisms are not identified |

### Claims the data cannot support

- coordinate-free, direction-invariant, or rotation-invariant transport;
- portable update amplitude;
- linear two-wall composition;
- innateness or universal hippocampal grammar;
- behavior independence;
- signed-displacement coding independent of distance;
- a pooled wall-and-reward replication statistic;
- priority language such as “first demonstration.”

## Recommended manuscript structure

1. **Known relative coding and the narrower update question.** Establish that
   boundary-, reward-, and context-relative representations are prior facts.
2. **Common logical framework.** Define registered-cell change vectors without
   implying one shared estimator across datasets.
3. **Exact-seam recurrence.** Present as validation/prerequisite, not the main
   discovery.
4. **Cross-location wall prediction.** Lead with held-out non-overlapping source
   and target seams.
5. **Exact versus shifted.** Make the decrement the defining result and the
   paired animal plot visually central.
6. **Spatial specificity and positive-model adjudication.** Put the full
   alternative-source distribution and model comparison here if feasible;
   retain the 3/6 tangential and 5-animal mirror limitations.
7. **Reward transition-class convergence.** Keep the full robustness battery
   concise in the main text and move exhaustive variants to supplement.
8. **Behavior-shared geometry.** Visually separate original neural S,
   behavior-only S, residual neural S, and held-out trial MSE improvement.
9. **Convergence without equivalence.** State that the arms differ in geometry,
   estimator, support, and identifiability.
10. **Discussion.** Open with what transferred and what did not; then positive
    models, behavior, dataset-specific limits, and a counterbalanced prospective
    experiment.

The active manuscript has no significance statement by explicit author
decision (`fa84fe5`). The requested quantitative claim-source artifact should
therefore map every abstract sentence and record `significance_statement` as
absent, rather than silently reintroducing one.

## Decision after audit

A positive-model protocol is scientifically necessary and can be specified
without selecting the endpoint by observed neural performance. Feasibility of
executing it depends on whether the released wall files expose enough
training-only, cell-resolved maps and geometry metadata to construct M0/M1 and
score the pre-existing target queries without leakage. That assessment must be
made only after the protocol is committed. If those source arrays are absent,
the stop condition applies: retain descriptive interpretation A, report B as
unresolved, and specify the additional cell-by-bin/session data required.
