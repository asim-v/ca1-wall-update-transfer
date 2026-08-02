# Positive-model adjudication protocol

Protocol version: 1.0

Protocol date: 2026-08-02

Status at commit: **frozen before any new positive-model target endpoint is
calculated or inspected**

This protocol was written after the descriptive-result audit in
`CA1_SPECIFICITY_AND_POSITIONING_AUDIT.md`. The already published exploratory
wall correlations and frozen-local reward results were known. No result from
the new M0–M3, empirical-rank, or R0–R4 endpoints was available when this
protocol was committed.

## Question and interpretation boundary

The primary reviewer objection is that the cross-location correlations are an
expected consequence of stable boundary-relative or reward-relative tuning,
generic spatial smoothness, and measured behavior. Permutation nulls do not
answer that objection. The new question is therefore predictive and nested:

> Does the empirical registered-cell source update add held-out predictive
> information beyond a predeclared positive spatial/reference-frame model?

A positive incremental result will justify only:

> Partial cross-location transfer is not exhausted by the tested positive
> model.

It will not disprove all boundary-vector, reward-relative, behavioral, or
smooth-place explanations. A null incremental result will be reported as a
mechanistic bridge—stable relative tuning is sufficient to generate the
observed update geometry—not suppressed as a failed preferred hypothesis. An
underpowered or heterogeneous result will leave the descriptive partial-
transfer claim intact and model-level reuse unresolved.

## General rules

1. The biological unit is the animal.
2. Cells, bins, seams, queries, sessions, source pairs, and permutations are
   estimator internals and support counts, never independent replicates.
3. Eligibility is determined without the held-out target neural outcome.
4. Model complexity and all numerical hyperparameters are selected inside
   training data. No target score may choose a model family, feature, penalty,
   transformation, or matching tier.
5. Target queries are inherited unchanged from tracked query ledgers; no query
   may be added or removed based on a new endpoint.
6. All planned models and outcomes, including failures and negative values, are
   written to the result artifact.
7. The wall analysis remains exploratory. “Frozen-local” below means frozen by
   this repository commit, not prospectively preregistered before data
   collection or before every prior exploratory analysis.

## Part I: boundary positive-model adjudication

### Frozen cohort and target ledger

- Dataset: Lee et al. (2025) longitudinal dorsal-CA1 geometric-deformation
  dataset.
- Animals: the seven released mice already used by the cross-location analysis.
- Target ledger: exactly the 405 primary one-grid-step same-signed-normal
  geometry- and support-eligible target queries among the 415 enumerated query
  records in `results/source_data/boundary_fragment_cross_location_transfer.json`
  at protocol time.
- Ledger artifact SHA-256:
  `6B20DA7167018F8B31700E16996E2C2CCE60587833FBF5EC1100B4E951BA1CC0`.
- Primary rate mode: `global_rate_demeaned`.
- Sensitivity rate mode: `raw_local_rate`.
- Primary source relation: a distinct non-overlapping oriented seam whose
  midpoint is exactly one 25 cm lattice step from the target and whose wall has
  the same signed normal.
- Existing ledger scale: 405 target queries and 658 eligible source pairs as
  documented by the current manuscript. The implementation must recompute and
  assert these counts from the frozen ledger before scoring. A mismatch is a
  stopping error, not a reason to update the expected counts silently.

The target wall label, geometry, occupancy, and cross-session registration may
be used for query construction and support. The target neural rate vector may
be used only after all training, hyperparameter selection, and target-specific
predictions are complete.

### Source, target, cell, and bin support

For every inherited target query:

- the source and target local strips must be spatially non-overlapping;
- source midpoint distance is exactly 25 cm in the primary analysis;
- both source wall and source open training states must exist;
- target evaluation uses the target wall exposure minus its later familiar
  square;
- source templates use the earlier training cycle and its preceding familiar
  square;
- training and target evaluation neural sessions are disjoint;
- the registered-cell intersection is the existing target-query intersection
  across target test, later square, training square, and all required
  non-target training sessions;
- a query requires at least 20 common registered cells;
- a location requires at least six common 5 cm bins with at least 0.5 seconds
  occupancy per contributing session;
- the identical target cell vector and spatial support are forced across M0,
  M1, M2, M3, and empirical alternatives.

If a fitted model cannot be evaluated on exactly that inherited common support,
the query is marked model-missing for every model. It is not retained for a
favorable subset of models.

### Outer holdout and leakage barrier

The atomic outer test object is one inherited target query. The complete target
wall neural session and its later-square neural baseline are withheld from:

- feature scaling;
- ridge selection;
- per-cell coefficient fitting;
- low-level reliability estimation;
- source weighting;
- stacking/amplitude calibration;
- missing-value imputation.

Where multiple queries share a target exposure pair, they are treated as one
outer fold for fitting and hyperparameter selection: **leave one target exposure
pair out**. This prevents another strip from the same neural maps leaking into
the fit. All query scores from that exposure pair are computed only after the
fold model is frozen.

Training-only pseudoqueries may be built from the remaining exposure pairs with
the same geometry rules. Pseudoquery outcomes may be used for nested
cross-validation and stacking because they are not part of the held-out target
exposure pair. A pseudoquery may not reuse either neural session in the outer
test pair.

### Prediction target

For cell `c` and target query `q`, the evaluation vector is the existing local
target residual:

`y[c,q] = target-wall local rate - later-square local rate`,

aggregated across the frozen local strip bins exactly as in the existing
cross-location analysis. Predictions from every model are cell vectors on the
same support and in the same rate units before correlation scoring.

### Frozen model families

The purpose is a strong, interpretable baseline, not an unrestricted model zoo.
No neural network, Gaussian process, arbitrary latent dimension sweep, or
post-hoc feature expansion is permitted in version 1.0.

#### M0: spatial/place-field baseline

M0 predicts each cell's target residual from stable absolute spatial structure
without an internal-boundary source update. Its training design matrix contains:

1. an intercept;
2. the cell's familiar-square rate at the same 5 cm bin, obtained only from the
   training-side familiar square available in the outer fold;
3. allocentric x and y represented by cubic B-spline bases with fixed knots at
   the outer-arena quartiles;
4. their tensor interaction restricted to products of the non-intercept x and y
   spline terms;
5. distance to the four fixed outer boundaries, linearly capped at 25 cm.

The familiar-square rate is a per-cell covariate; geometry terms allow the
mapping from that covariate to residual activity to vary smoothly over absolute
position. No target neural map supplies a familiar-square covariate: if only the
held-out later square is available, M0 uses the training-side square and the
query becomes missing if registration/support cannot be aligned.

#### M1: boundary-relative positive model

M1 contains every M0 feature plus this fixed internal-boundary set, computed
from geometry alone at each accessible 5 cm bin:

1. distance along each of the four allocentric cardinal rays to the first
   internal boundary, capped at 25 cm, with a separate “none within 25 cm”
   indicator;
2. nearest internal-boundary distance, capped at 25 cm;
3. one-hot signed normal of that nearest boundary (`+x`, `-x`, `+y`, `-y`);
4. interactions of nearest distance with its signed-normal indicators;
5. local wall/open indicators for the four grid-neighbor seams of the containing
   accessible partition;
6. number of blocked adjacent partitions within one lattice step;
7. fraction of the 3×3 arena that is accessible.

These features capture canonical distance/direction tuning, local topology, and
coarse global geometry without adding arbitrary shape identities. Wall presence
and inaccessible area remain experimentally confounded; M1 is therefore a
boundary/topology model, not a pure visual-wall model.

M0 and M1 are fit separately for each cell by ridge regression on all eligible
training environment-bin observations in the outer fold. No cell-selection
threshold based on target response is allowed.

#### M2: empirical source-update model

M2 uses the existing source wall-minus-open registered-cell vector. If several
same-relation source seams are eligible for a query, their vectors are averaged
with equal weight, exactly as in the existing estimator. No reliability or
neural-similarity weighting is allowed in the primary analysis.

For correlation scoring, M2 is used directly after subtracting its cell mean;
amplitude is irrelevant. For MSE scoring, a single scalar amplitude and
intercept are learned from training-only pseudoqueries in the outer fold.

#### M3: combined positive plus empirical-update model

M3 adds to the M1 prediction the component of M2 not predicted by the positive
model at the source relation:

`M3 = M1_target + beta * (M2 - M1_source_effect)`.

`M1_source_effect` is the M1-predicted source wall-minus-open difference,
computed without the outer target neural pair. `beta` and a stack intercept are
fit using training-only pseudoqueries. This formulation asks whether the
empirical source contains an incremental cell-specific component rather than
crediting M3 for a boundary effect already captured by M1.

If training pseudoqueries cannot identify `beta` under the gates below, M3 is
not scored and the positive-model adjudication is declared infeasible. `beta`
must not be set from the target outcome or replaced post hoc by a convenient
constant.

### Hyperparameter selection

All continuous features are centered and scaled using the outer-fold training
observations only. Binary indicators are not scaled. Constant columns are
dropped based only on training data and logged.

M0 and M1 ridge penalties are selected separately from the fixed grid

`lambda / mean(diag(X'X)) ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}`.

Selection uses leave-one-training-environment-pair-out nested cross-validation.
The loss is the median across cells of normalized bin-level MSE, then averaged
equally across held-out training environment pairs. Ties within `1e-12` select
the larger penalty. The same chosen penalty is used for all cells in an animal
and outer fold.

M2 amplitude and M3 `beta` use ridge stacking with penalty grid
`{0, 0.01, 0.1, 1, 10, 100}` after both component predictions are scaled to
unit root-mean-square magnitude using training pseudoqueries. Selection uses the
same nested folds and loss. Ties select the larger penalty. Stack coefficients
may have either sign; their signs are reported. No non-negativity constraint is
imposed because an imposed positive coefficient would bias the incremental
test.

### Primary endpoint and aggregation

For each query and model, compute Spearman correlation across the common
registered-cell vector between prediction and held-out `y`. A query with fewer
than 20 finite paired cells after model prediction is model-missing for all
models.

The frozen primary contrast is

`Delta_q = Spearman(M3_q, y_q) - Spearman(M1_q, y_q)`.

Aggregation is:

1. equal-weight eligible same-relation source seams within target query;
2. equal-weight target queries within target exposure pair;
3. equal-weight target exposure pairs within animal;
4. equal-weight animals for the cohort descriptive mean.

The sign convention is positive when adding the empirical source improves
prediction. The primary inferential test is the exact one-sided sign-flip
distribution of the seven animal-level `Delta` values. Report the observed
mean, median, every animal value, positive-animal count, all 128 sign-flip
means, the plus-observed upper-tail fraction, and leave-one-animal-out means.
The two-sided exact tail is also reported. There is no cell- or query-level
population P value.

Primary adjudication is called positive only if:

- all seven animals retain an estimable M1 and M3 animal summary;
- the cohort mean `Delta` is positive;
- at least six of seven animal values are positive; and
- the exact one-sided animal sign-flip P is at most 0.05.

The numerical effect and uncertainty remain primary; this gate is only a
predeclared interpretation label.

### Planned secondary outcomes

All are reported regardless of the primary result:

1. M0, M1, M2, and M3 held-out Spearman correlations by animal.
2. M1-minus-M0 and M3-minus-M2 correlations.
3. M1-to-M3 change in normalized held-out MSE, with positive meaning lower MSE.
4. Cross-fitted residual alignment between `y - M1_target` and
   `M2 - M1_source_effect`.
5. Raw-rate sensitivity using the identical folds and hyperparameters selected
   independently within raw-rate training data.
6. Model performance versus the exact-location predictor on the inherited
   exact-versus-shifted query subset.
7. Coverage ledger by animal, exposure pair, query, cells, bins, and failure
   reason.

No transfer fraction will be introduced in version 1.0. It would require a
separate frozen paired estimator and denominator-stability audit.

## Part II: fully enumerated empirical spatial-continuity baseline

This control is frozen independently of M0–M3 and must be executed if the
existing source-query records contain all candidate source vectors, even if the
fitted model is infeasible.

### Candidate enumeration

For each frozen target query, enumerate every source seam that meets all of:

1. source midpoint exactly 25 cm from target midpoint;
2. non-overlapping source and target strips;
3. source has both wall and open training states;
4. source training sessions are disjoint from the target evaluation pair;
5. identical target-query registered cells and bins can be used;
6. at least 20 cells and six bins under the occupancy gates above;
7. source seam is not the target seam;
8. no neural similarity, correlation, or endpoint enters enumeration.

The “correct relation” is the pre-existing same-signed-normal class. Every
other signed normal at the same distance is an alternative. Sources are not
cherry-picked within either class.

### Frozen matching hierarchy

Only these tiers may be reported, in this order; failure at a tier is not a
license to invent a relaxed tier.

- **Tier 1 (primary):** exact 25 cm midpoint distance and all gates above.
- **Tier 2 (strict tangential sensitivity):** Tier 1 plus translation vector
  perpendicular to the wall normal (translation along the wall axis).
- **Tier 3 (facing-overlap sensitivity):** Tier 1 plus the existing documented
  facing/away and strip-overlap classifications, reported separately.

No nearest-distance approximation, wider distance caliper, or post-hoc pooling
is allowed in version 1.0.

### Empirical-rank endpoints

For each target query, score every candidate source by Spearman correlation of
its wall-minus-open vector with the held-out target residual. Ties receive
midranks. Compute:

1. percentile rank of the equal-weight mean correct-relation score among all
   candidate-source scores;
2. correct-relation mean minus mean of all alternatives;
3. correct-relation mean minus the best alternative score;
4. indicator that the correct-relation mean exceeds every alternative.

Aggregate each outcome by equal-weight query, target exposure pair, and animal,
then use exact animal sign flips for signed contrasts. For percentile rank,
subtract 0.5 before sign flips. Report candidate counts and orientation counts
per query. The primary empirical endpoint is animal-mean centered percentile
rank in Tier 1. Tier 2 remains explicitly support-limited if fewer than all
seven animals are eligible.

## Part III: reward positive-model adjudication

The reward analysis is a convergent extension. It is not allowed to delay or
obscure the primary boundary adjudication.

### Frozen cohort and endpoint

- Eight confirmatory mice: `m3`, `m4`, `m7`, `m12`, `m13`, `m14`, `m15`,
  `m19`.
- Existing specificity definition and aggregation remain unchanged.
- Confirmatory result ledger SHA-256:
  `4C9A85B130B22E6E0428CD9F0F9D3586F62A89ED4A799ECB36D70A1AF1E1C47E`.
- Frozen nonlinear-behavior result ledger SHA-256:
  `36C2C813467F9158C47993D33F9640351DC97BC6FFE456247E68A67743855B12`.
- Neural signals: dF/F primary, deconvolved events robustness.
- Cell and registration gates remain those in the confirmatory scoring commit
  `b805c56` and registration lock `862e718`.

The design cannot identify displacement sign independently of distance: the
two repeated transitions are same-signed 120 cm moves and the comparison is an
opposite-signed 240 cm return. Every output must therefore say “transition
class,” not “signed-displacement operator.”

### Outer split

The outer neural holdout is one switch session. All trials in that session,
before and after the switch, are excluded from fitting cell-specific positive-
model coefficients and hyperparameters. Models are trained within animal and
visual environment using other registered switch sessions. Day 8 remains
excluded. A model requiring the held-out post-switch neural outcome to estimate
a reference-frame weight is invalid.

If a cell lacks sufficient cross-session registration or training trials, it is
missing for every compared reward model in that held-out session. At least 20
common registered cells and all three transition classes per visual environment
are required for an animal-level specificity value.

### Frozen reward model families

All models predict trial-level neural activity, from which held-out pre/post
maps and update fields are generated before calculating the existing S
statistic.

- **R0 position-only:** per-cell ridge-GAM with track-relative position (45
  circular 10 cm bins represented by fixed circular cubic splines), trial
  number, and training-only cell intercept/reliability.
- **R1 track plus reward reference frames:** R0 plus the same circular basis in
  position relative to current reward and a pre/post reward-state indicator.
  Track and reward terms are additive; no cell class is selected.
- **R2 behavior-conditioned:** corrected old-reward-frame speed and licking,
  their fixed quadratic terms, and speed-by-lick interaction, with trial number;
  no neural reference-frame terms beyond a cell intercept.
- **R3 combined:** union of R1 and R2 features with one ridge penalty.
- **R4 residual update:** held-out observed neural activity minus R3
  prediction; pre/post maps, update fields, and S are recomputed from these
  residuals.

R1–R3 use the same outer folds, trials, cells, and bins. Hyperparameters use
leave-one-training-session-out cross-validation and the fixed normalized ridge
grid specified for M0/M1. Loss is Poisson deviance for nonnegative events and
MSE for dF/F. If dF/F values or model implementation make a common design
matrix impossible, separate signal-specific fits are allowed but must use the
same feature family and folds.

### Reward primary endpoint

The frozen reward endpoint is the animal-level R4 residual transition-class
specificity S. The key positive-model comparison is observed neural S minus
R3-predicted S together with R4 S; both are reported because nonlinear
prediction and subtraction need not make the specificity statistic additive.

Aggregation remains environment within mouse, then equal-weight mice. Report
all animal values, positive count, exact one- and two-sided sign-flip P values,
mouse-bootstrap 95% interval, and leave-one-mouse-out means. A residual result
is called positive only if mean R4 S > 0, at least 7/8 mice are positive, and
the exact one-sided sign-flip P <= 0.05.

Planned performance diagnostics are held-out trial deviance/MSE for R0–R3,
predicted S for R0–R3, event-trace replication, and cell-coverage ledgers. A
model may explain transition specificity while explaining modest total trial
variance; both quantities must be shown and neither substitutes for the other.

## Missing data

- Missing neural values are never mean-imputed from target data.
- Training feature medians may impute continuous geometry/behavior values only
  when missingness is below 5% within an outer training fold; the count and
  variable are logged. Above 5%, that model/fold fails.
- Missing binary geometry is a data error and fails the query.
- A model comparison uses the intersection of predictions from all compared
  models.
- Animal summaries are not imputed.
- No eligibility gate may depend on the sign or magnitude of a new neural
  endpoint.

## Failure and stopping conditions

Stop before scoring a new neural endpoint, or terminate execution without
interpreting a partial endpoint, if any of these occurs:

1. the released data do not contain cell-resolved training and target maps,
   geometry, occupancy, and registration needed for the declared split;
2. the inherited target ledger cannot be reconstructed with the expected
   counts/hash semantics;
3. target neural maps would be required for feature scaling, hyperparameter
   choice, model fitting, reliability weighting, or query selection;
4. M3 stacking cannot be identified from training-only pseudoqueries;
5. fewer than all seven wall animals have an estimable primary M1–M3 contrast;
6. exact Tier-1 alternative matching yields fewer than five wall animals (the
   rank analysis stops; the fitted analysis may continue independently);
7. reward R0–R4 cannot be trained with a whole switch session held out, or the
   three-transition structure leaves fewer than six estimable reward animals;
8. code tests fail, result regeneration is nondeterministic beyond floating-
   point tolerance, or a tracked source hash changes unexpectedly;
9. positive-model complexity can only be chosen after viewing held-out target
   performance.

When a stop occurs, write the full support/failure ledger and state exactly
which data are missing. Do not relax gates, change the endpoint, or inspect a
partially computed target result to rescue power.

## Planned result artifacts and tests

If feasibility gates pass, implementation must produce:

- `results/positive_model_boundary_v1.json`
- `results/positive_model_spatial_rank_v1.json`
- optionally `project2-goal-update-transfer/results/positive_model_reward_v1.json`
  if the reward split passes independently;
- `POSITIVE_MODEL_ADJUDICATION_RESULTS.md`, containing all planned outcomes;
- deterministic unit tests for geometry features, fold isolation, ridge
  selection, stacking, aggregation, sign-flip inference, source enumeration,
  and missingness gates;
- explicit assertions that target sessions never occur in a training fold;
- a machine-readable protocol metadata block with this protocol commit hash,
  source hashes, code commit hash, random seeds, library versions, and run time.

Random operations use NumPy `PCG64` seed `20260802`. Exact enumerations do not
use Monte Carlo. Floating-point comparisons in tests use `rtol=1e-10` and
`atol=1e-12` unless a library operation documents weaker deterministic
precision.

## Frozen interpretation table

| Outcome | Permitted conclusion | Forbidden conclusion |
|---|---|---|
| M3 improves on M1 | Empirical source updates contain held-out information not exhausted by this tested spatial/boundary model | All boundary-vector or smooth-place accounts are disproved |
| M1 matches M3 and explains transfer | Stable relative tuning is sufficient to generate reproducible cell-resolved update geometry under this test | There is no structured reuse or the descriptive correlation was false |
| Result heterogeneous/underpowered | Descriptive partial transfer remains; positive-model adjudication is unresolved | Absence of evidence proves no model-level reuse |
| Correct relation ranks above full alternatives | The observed relation is more specific than the enumerated matched spatial alternatives | Coordinate-free transport or general direction invariance |
| Reward R4 remains positive | Transition-class geometry is not exhausted by the tested reference-frame/behavior model | Behavior independence or an identified signed-displacement operator |
| Reward R3 predicts observed S and R4 is null | Conventional reference-frame and measured-behavior structure can account for the update alignment tested here | All CA1 reward remapping is behavior or no neural result exists |
