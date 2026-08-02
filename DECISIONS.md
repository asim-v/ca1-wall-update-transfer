# Decision Log

Use one entry per consequential scope or methodological decision.

## 2026-07-28 — Select boundary-normal CA1 metric as primary project

**Status:** Accepted

**Decision:** Lead with local boundary-normal versus tangential metric
anisotropy in the Lee et al. geometric-deformation dataset.

**Why:** It combines a precise directional prediction, manageable public data,
first-derivative geometry, a learning component, and direct comparison with
existing cognitive-map models.

**Alternatives considered:** Fisher-Rao audit of CA1 hyperbolicity; cue-locked
population-trajectory bending; reward-relative compositional isometry.

## 2026-07-28 — Do not lead with pointwise curvature

**Status:** Accepted

**Decision:** Treat the pullback metric and anisotropy as confirmatory.
Gaussian/Ricci curvature remains exploratory.

**Why:** Pointwise curvature requires unstable second derivatives and is
especially vulnerable to smoothing and arena-boundary artifacts.

## 2026-07-28 — Use a cross-validated deterministic pullback metric

**Status:** Accepted

**Decision:** Use cross-products of Jacobians independently estimated in
disjoint temporal-block folds as the confirmatory metric. Use pooled
\(g=J_F^\top J_F\) only for visualization. Treat a Poisson-Fisher metric as a
sensitivity analysis.

**Why:** The dataset contains calcium-derived activity rather than literal
Poisson spike counts, and plug-in derivative products have positive
noise-dependent bias.

## 2026-07-29 — Narrow the causal and novelty claim

**Status:** Accepted

**Decision:** Analyze newly introduced boundaries on their shared accessible
side. Do not frame reverse transitions as isolated boundary removal. Explicitly
distinguish within-condition metric stretch from the paper's published
between-condition correspondence displacement.

**Why:** Deformations occlude entire 25 × 25 cm partitions, changing accessible
space together with boundaries. Figure 6 already contains directional
population-vector flow fields.

## 2026-07-29 — Replace inert cell-label permutation

**Status:** Accepted

**Decision:** Use cell-wise spatial map shifts/rotations, stationary-code
trajectory replay, matched pseudo-boundaries, and trace-position shifts.

**Why:** Neuron relabeling leaves
\(N^{-1}\sum_i\nabla F_i\nabla F_i^\top\) exactly unchanged.

## 2026-07-29 — Freeze the first biological pilot before target inspection

**Status:** Accepted

**Decision:** Use QLAK-CA1-51 and the center-occlusion (`o`, blocked partition
4) condition. For sequence 1, compare session 2 with the average metric from
its bracketing square sessions 1 and 11. Repeat the same analysis for sequence
2 (sessions 12 versus 11 and 21) as a preliminary experience contrast. Use the
four wall midpoints, normal distances 2.5, 7.5, and 12.5 cm, four contiguous
temporal folds, and bandwidths 5, 7.5, and 10 cm. The raw common-cell event
code is primary; coarse occupancy-balanced weights and a five-session stable
cell core are sensitivity analyses.

**Why:** The center occlusion supplies four equal-length internal boundary
segments and paired accessible sides without selecting a wall based on its
neural outcome. Bracketing squares reduce monotonic recording-drift bias.
Animal 51 is the smallest downloadable file and has two complete sequences,
making it appropriate for pipeline validation but not population inference.

## 2026-07-29 — Amend pilot after a support-gate failure

**Status:** Accepted for diagnostics; not a confirmatory result

**Decision:** Retain the frozen condition, distances, bandwidths, folds, and
quality thresholds, but replace one exact midpoint per segment with two
predefined interior points at 40% and 60% of segment length. Report the
original cross-metric normalization and, separately, a pooled-square metric
used only as a positive scale for the cross-validated contrast numerator.

**Why:** The first execution left only 0–9 of 12 queries valid depending on
sequence and bandwidth, and cross-validated square traces were sometimes
non-positive, making ratios undefined or extremely unstable. Expanding
interior coverage is based on behavioral support rather than neural effect
size. The pooled denominator does not change the numerator sign, but it must
remain explicitly labeled because it contains positive derivative-noise bias.

## 2026-07-29 — Withdraw the broad novelty claim

**Status:** Accepted

**Decision:** Do not claim the first CA1 Jacobian analysis, differential-
geometric analysis, boundary-normal sensitivity result, or generic
experience-dependent stabilization. Frame the contribution as a
crossvalidated, longitudinal true-wall-versus-pseudo-wall residual in an
introduced-internal-wall paradigm.

**Why:** Wang, Monaco, and Knierim (2020) explicitly calculated the local
firing-rate-map Jacobian and its maximal-change direction, finding preferential
directions perpendicular to texture boundaries. Tanni, de Cothi, and Barry
(2022) found narrower fields normal to walls and faster population change on
orthogonal than parallel runs. Lee et al. already reported general reliability
increases with experience in the target dataset.

## 2026-07-29 — Treat animal 51 as estimator-development data

**Status:** Accepted

**Decision:** Do not use QLAK-CA1-51 for confirmatory biological inference.
Use it to diagnose support, construct folds, and validate synthetic trajectory
replay. Freeze the redesigned estimator using position-only and synthetic
criteria before inspecting neural outcomes in another animal.

**Why:** Contiguous-quarter folds produced sparse query coverage, weak
split-pair reliability, threshold-fragile estimates, and square-square changes
comparable to the target contrast. Two-fold estimates increased coverage but
removed the reliability diagnostic and changed signs across bandwidths.

## 2026-07-29 — Freeze the untouched-animal estimator

**Status:** Accepted before inspecting QLAK-CA1-08 neural outcomes

**Decision:** For the center-occlusion analysis in QLAK-CA1-08:

- use the 48-cell core registered across all four square and three `o`
  sessions;
- assign whole 60-second blocks to four folds using position-only 5 cm
  occupancy balance, excluding a 1-second guard at each block edge;
- apply the center-partition mask and coarse cross-session occupancy weights
  identically to condition and bracketing squares;
- query each of four wall segments at 40%, 50%, and 60% of its length and at
  normal distances 2.5, 7.5, and 12.5 cm;
- use a 10 cm tricube bandwidth as primary and 7.5 cm as a localization
  sensitivity; exclude 5 cm from biological inference because it has no
  immediate-wall support in trajectory replay;
- use the cross-fold tensor contrast as the numerator and a clearly labeled
  pooled-square tensor only as a positive display scale;
- assess reliability from \(g_{nn}\), \(g_{tt}\), and their residual contrast
  across fold pairs `(0,1)` and `(2,3)`, not from flattened tensor entries.

The primary sequence-level gate requires at least four valid near-wall queries
spanning at least three wall segments, positive normal magnification and
normal-minus-tangent contrast at 10 cm, contrast reliability above 0.3, and a
target contrast larger in magnitude than the square-to-square pseudo-wall
change. The 7.5 cm sign must agree when it has adequate support.

**Why:** On QLAK-CA1-51 trajectories, a known normal warp recovered the correct
sign at 7.5 and 10 cm, decayed with distance, and had fold-pair contrast
reliability of 0.99 and 0.98, respectively. The matched flat null was near zero
with much lower reliability. Ten centimeters retained 27/36 queries, including
6/12 immediately adjacent queries, while respecting segment endpoints at the
frozen tangential anchors.

## 2026-07-29 — Behavior-select the plus deformation as a held-out follow-up

**Status:** Accepted before loading `+`-session neural traces

**Decision:** The center-occlusion analysis failed in QLAK-CA1-08. Audit all
nine repeated deformations using only positions, masks, and foldwise design
quality. Carry forward the top-ranked `+` deformation (sessions 6, 16, and 26,
bracketed by squares) with the same estimator and the 47-cell seven-session
core. Because it has eight wall segments, require at least six valid near-wall
queries spanning at least four segments, in addition to the previously frozen
sign, reliability, and pseudo-wall gates.

**Why:** Query-balanced 10 cm support for `+` remained strong across all three
sequences: 21, 11, and 9 near-wall queries spanning 8, 6, and 5 segments.
Center occlusion ranked eighth and had 5, 3, and 0 near-wall queries. The
selection used no neural response values, but it is still a post-design
condition selection in an already opened animal and must be labeled an
exploratory held-out-condition test, not cohort confirmation.

## 2026-07-29 — Freeze the first independent-animal replication

**Status:** Accepted after the QLAK-CA1-08 `+` result and before downloading or
loading QLAK-CA1-74

**Decision:** Use QLAK-CA1-74, the smallest unopened archive, as the first
independent-animal replication. Inspect only session labels and positions
first. Analyze the `+` condition only if all three sequences contain at least
six design-supported near-wall queries spanning at least four wall segments at
the primary 10 cm bandwidth. If eligible, use the unchanged 60-second
query-balanced folds, 1-second guards, 10 cm primary and 7.5 cm sensitivity,
seven-session stable-cell core, square pseudo-wall null, and all previously
frozen neural gates. Do not replace a failed sequence with another deformation.

**Why:** At the time of this decision, the provisional coarse-fold output made
the second QLAK-CA1-08 `+` exposure appear to pass every gate. The later
implementation audit invalidated that output: under the intended
query-balanced folds, all three anisotropy estimates are positive but all
three fail the reliability gate. The decision to seek independent animals is
still appropriate, but its original trigger and the corrected result are both
recorded. Animal selection by unopened file size is independent of neural
response, and a positions-only eligibility gate prevents low-coverage neural
estimates from being interpreted after the fact.

**Prospective amendment after the QLAK-CA1-74 behavior-only exclusion:** Screen
the remaining unopened archives in ascending file-size order (56, 30, 50, 75)
using the identical positions-only `+` eligibility gate. Neural data will be
loaded for every eligible animal and for no ineligible animal. This changes
only the replication sampling plan; the condition, estimator, and outcome
gates remain frozen.

**Implementation correction after first viewing QLAK-CA1-56:** A code audit
found that `audit_all_condition_support.py` used the intended position-only
query-design-balanced folds, while `run_frozen_center_boundary.py` still called
the older coarse-occupancy-balanced fold function. Invalidate the existing
QLAK-CA1-08 and QLAK-CA1-56 output files, replace that call with the already
tested query-balanced function at the frozen 10 cm design bandwidth, and rerun
both animals. No query, neural gate, effect threshold, or sensitivity setting
changes. This is an implementation correction, not a new analysis choice; it
is recorded after seeing the provisional animal-56 result.

**Frozen cohort table before loading QLAK-CA1-30 or QLAK-CA1-50 traces:**
Animals 30 and 50 pass; animals 74 and 75 fail; animal 56 passes the behavioral
screen and retains its implementation-correction caveat. Analyze all eligible
sequences regardless of subsequent neural reliability or sign. The exact
support counts and roles are recorded in
`reports/plus_cohort_eligibility.md`.

## 2026-07-29 — Retain the directional result; do not support stabilization

**Status:** Accepted after completing the eligible `+` cohort and explicitly
post-outcome controls

**Decision:** Retain as a promising descriptive result that introduced
internal walls are associated with a preferentially boundary-normal CA1
response-tensor change. Do not claim that the frozen confirmatory reliability
criterion passed, that experience stabilizes the geometry, that the effect is
caused by a boundary independently of occluded space, or that boundary removal
was tested.

**Why:** At 10 cm, all 12 eligible animal-by-exposure estimates had positive
normal magnification, positive normal-minus-tangent contrast, and a contrast
larger than the absolute frozen square-session drift comparator. This includes all
six exposures in the two clean animals prospectively locked at commit
`443d038`.
However, only 3/12 sequences passed every frozen gate because the query-pattern
fold-pair reliability usually fell below 0.3.

On identical longitudinal query support, all 12 contrasts remained positive,
but exposure-3-minus-1 effect size increased in two animals and decreased in
two. Reliability increased in three animals and decreased in one, with
nonmonotonic trajectories. An exact segment-label calibration, staged circular
trace-position nulls in the two clean animals, an equal-47-neuron sensitivity,
and additive speed/movement-direction adjustment all preserved the directional
signal.
Those diagnostics were specified after outcomes and strengthen robustness
without upgrading confirmatory status.

## 2026-07-29 - Proceed with a bounded reanalysis manuscript

**Status:** Accepted

**Decision:** Prepare a working computational reanalysis paper that leads with
the consistent aggregate boundary-normal direction but treats the failed
query-pattern reliability gate and heterogeneous experience trajectories as
co-primary conclusions. Use "prospectively locked in version control" for
animals 30 and 50, retain the exploratory and implementation-correction roles
for animals 08 and 56, and label every post-outcome diagnostic explicitly.

**Why:** All 12 eligible exposures, including all six clean prospectively
locked exposures, show positive normal magnification, positive
normal-minus-tangent contrast, and target magnitude above the frozen
square-session drift comparator. That descriptive regularity is worth
documenting, but
only 3/12 exposures passed every frozen gate and common-support longitudinal
changes do not support stabilization. A transparent mixed-result paper is more
informative than either suppressing the directional observation or presenting
it as confirmed.

## 2026-07-29 - Reclassify the frozen square control as a drift comparator

**Status:** Accepted after final implementation audit

**Decision:** Preserve the frozen primary outputs, but do not describe their
square comparator as mask-matched or as a sham. Report target and square
near-query counts separately, state that the target used triplet-valid support
while the comparator used square-pair-valid support, and interpret the
comparison as square-session drift across the complete intervening
nine-deformation sequence. A same-mask rerun, if performed, is a post-outcome
sensitivity and cannot replace the frozen result.

**Why:** The frozen implementation uses
`pre.valid & condition.valid & post.valid` for the target and
`pre.valid & post.valid` for the square comparator. Near-query counts differ in
7 of 12 exposures by one to five queries. The all-12 exceedance remains exactly
reported for the frozen calculation, but it is not a same-sample causal wall
contrast or a randomized wall sham.

## 2026-07-30 — Suspend manuscript-first work and open a discovery campaign

**Status:** Accepted

**Decision:** Treat the boundary-normal manuscript as a transparent archived
mixed result. Search the released raw data for a simpler contribution, require
cheap falsification before implementation, and retain every failed branch in a
discovery ledger.

**Why:** The directional tensor aggregate is not a reliable local mechanism,
and experience does not stabilize it. More mathematical elaboration would not
repair that identification problem.

## 2026-07-30 — Correct the raw image-coordinate convention

**Status:** Accepted; stored-map wall-strip values unchanged

**Decision:** Use raw trajectory \(y\) directly in the paper's north-to-south
partition-row order. Remove the reflected partition centers and released-row
flip from seam helpers; correct introduced-boundary normals and accessible
support; add asymmetric-arena regression tests.

**Why:** In asymmetric shapes, blocked paper-row-zero partitions have no
samples at raw \(y<25\), proving that raw \(y\), stored-map rows, and paper
partition rows share image-style north-to-south order. The previous seam strip
landed in the correct stored bins only because a reflected raw frame and map
row flip canceled. Raw trajectory analyses did not have that protection.

## 2026-07-30 — Promote the exact-location wall-conditioned profile to provisional lead

**Status:** Accepted pending raw-event and kinematic gates

**Decision:** Lead the discovery track with recurrence of the exact-location
wall-conditioned cell profile across global geometries. Require both a
within-global-pair wall/wall versus
wall/open contrast and target-rate-held-out local prediction using independent
outer square baselines. The target wall label, occupancy, and registration
support are test-aware. Treat cells, seams, and permutations as estimator
samples; mice are the biological units.

**Why:** The corrected stored-map result is positive in 7/7 mice, survives
cell-ID permutation, global-rate removal, occupancy weighting, horizontal and
vertical subsets, and is localized within 10 cm of the wall. A profile learned
from other shapes beats an exact-location open-seam profile against a target
local cell vector in a future cycle in 7/7 mice. This is a cell-resolved
exact-seam consequence of the source paper's boundary-remapping result, not
blind global-shape prediction or evidence for a portable wall primitive.

## 2026-07-30 — Do not promote the other discovery screens

**Status:** Accepted

**Decision:** Stop the alias-resolution, blocked-field relocation/suppression,
first-encounter recruitment, acute impossible-transition, sequence
anticipation, imaging-plane microtopography, and cue-gradient branches.

**Why:** Each failed a specific robustness or identification gate. Full-map
alias effects collapsed under independent halves and displacement controls;
field suppression vanished under prospective selection; matched later visits
outpredicted first encounters in 6/7; acute wall responses followed
unmatchable turning; and the remaining effects were heterogeneous or already
explained by general reliability.

## 2026-07-30 — Pass the non-overlapping raw-event gate

**Status:** Accepted

**Decision:** Retain the exact-location wall-conditioned profile as the
provisional discovery lead after reconstructing every map from raw positions
and event traces split
into contiguous session halves. Treat the four crossed half assignments as a
symmetric estimator, not as four independent replicates, and retain the
12-environment-pair coverage gate.

**Why:** The wall/open advantage is positive in all 14/14 eligible exposure
blocks and all 5/5 eligible mice. A stricter dwell threshold retains 10/10
eligible blocks in 4/4 mice. Recombined raw halves reproduce released rates
and occupancy to machine precision after applying the authoritative
blocked-tile rate mask, with zero finite-mask mismatches.

## 2026-07-30 — Pass the additive kinematic-adjustment gate

**Status:** Accepted for the static observational result

**Decision:** Retain the exact-location wall-conditioned profile after
estimating raw-event 5 cm bin effects jointly with speed, allocentric
movement-direction harmonics, and linear session time. Evaluate every session
at one common physical reference calibrated only from the first familiar
square. Do not describe the covariates as head direction.

**Why:** The target-rate-held-out local advantage remains positive in 7/7 mice
with session time (animal mean 0.209) and without it (mean 0.210), with no sign
flips. Released-map reconstruction is exact, occupancy error is
\(5.7\times10^{-14}\) seconds, and adjusted finite masks match. This is an
additive linear nuisance control, not local kinematic matching; target-session
neural outcomes estimate that session's nuisance coefficients before scoring
but never enter the training profiles.

## 2026-07-30 — Freeze the context-matched focal-seam test

**Status:** Accepted before inspecting the context-matched neural outcome

**Decision:** Test whether the exact-location wall result survives after
holding the remainder of the accessible target partition's local boundary
context fixed. For each target-rate-held-out wall query, encode the blocked
versus accessible state of every grid-neighbor partition incident to the
accessible target partition, excluding the focal source partition. Retain
training wall and open environments only when this non-focal context exactly
matches the target environment. Exclude the target geometry's neural rates,
recompute common registered cells and spatial support after filtering, and
retain the existing cross-exposure square baselines, strip, dwell, cell, and
rate-mode definitions.

The primary outcome is the animal-mean matching-wall minus matching-open
Spearman correlation after removing each cell/session's arena-wide rate.
Report raw rates, horizontal and vertical subsets, exposure-pair support, and
every animal regardless of sign. Treat animals as biological units. This is
an exploratory strengthening test in an already analyzed dataset, not a
prospective confirmation.

**Why:** The current effect could reflect recurrence of the entire local
partition configuration rather than an independently editable wall segment.
The public design contains 246 target-rate-held-out records in which both a
wall and an open training profile can be formed while exactly matching the
target's other neighboring partition states, with at least nine records and
both wall orientations represented in every mouse. This contrast changes the
focal edge state while holding the immediately adjacent non-focal context
fixed; global shape and inaccessible area elsewhere can still differ.

**Implementation note after execution:** Recomputing cell and occupancy
eligibility after removing context-mismatched training sessions recovered six
queries that had failed the broader-profile support intersection. The tracked
context-matched artifact therefore contains 252 rather than 246 predictions.
This changes support, not the frozen context definition; the neural outcome
was positive in all seven mice.

## 2026-07-30 — Freeze the single-tile counterfactual transfer test

**Status:** Accepted before inspecting the counterfactual neural outcome

**Decision:** Within the context-matched target-rate-held-out design, further
restrict training profiles to wall/open environment pairs whose complete
blocked-partition vectors are identical after removing the focal source
partition. Thus the two training geometries differ only in whether that one
partition is accessible. Keep the target geometry excluded and require its
other target-neighbor states to match both training profiles. Use the same
primary global-rate-demeaned animal mean and report raw rates and all mice.

**Why:** Forty-six already eligible predictions across all seven mice admit
this exact geometric counterfactual, supplied by training environment `u`
(partitions 4 and 5 blocked) versus `o` (only partition 4 blocked), with
partition 5 as the focal source. If its profile difference transfers to a
different held-out geometry containing the same focal wall, it is a sharper
test of a reusable local structural edit than averaging unrelated global
shapes. The fixed environment order and the wall/inaccessible-tile
confounding remain.

## 2026-07-30 — Promote limited local-update reuse, not a portable wall code

**Status:** Accepted after spatial, identity, order, composition, learning, and
independent-dataset audits

**Decision:** Replace the exact-location-only lead with the narrower stronger
claim that a registered-cell wall-minus-open local update partially transfers
to a non-overlapping seam one 25 cm grid step away in the preselected
same-signed-normal relation. Report exact-location performance, translated
performance, and transport loss together. Require target wall-state
specificity and disclose the nonuniform coherent cell-identity diagnostic.
Do not call the effect a location-invariant wall primitive or general
direction-selective code.

**Why:** The exact local-context match remains positive in 7/7 animals (mean
advantage 0.184), and the `u-o` single-tile counterfactual remains positive in
7/7 (mean 0.225). At the shifted seam, the source effect predicts the held-out
target residual at mean \(r=0.148\), positive in 7/7, and favors the target
wall residual over its open profile by 0.216, positive in 7/7. The strips share
no spatial bins. This shifted estimator reconstructs a separate multi-geometry
source vector rather than transporting the `u-o` endpoint difference. Transfer
is weaker than its query-matched exact-location benchmark (\(r=0.230\)) by
0.082, and layout-matched opposite-direction comparisons do not support a
general direction claim. Coherent identity-null tails are strong in five
animals but not Q51 or Q75.

**Raw-event gate after execution:** Additive adjustment for speed, allocentric
movement direction, and session time retains cross-location prediction at mean
\(r=0.119\), positive in 7/7 animals. Specificity relative to the
training-derived target-open profile is 0.172, positive in 7/7. The audit
matches stored-map query/session/seam identities and common-cell/bin counts
across 405 target queries and 658 source pairs. Target-session outcomes
necessarily estimate that session's
nuisance coefficients before scoring, so this remains a nuisance-adjusted
evaluation rather than fully outcome-blind preprocessing.

## 2026-07-30 — Retain the fixed-order ceiling

**Status:** Accepted

**Decision:** Use the same-seam unchanged-state later-minus-earlier vector only
as a falsification of generic sequence drift. Do not interpret it as removal
of the causal order confound.

**Why:** In all seven animals, the actual `u-o` edit remains predictive after
subtracting generic order drift (mean \(r=0.212\)), whereas the generic order
vector alone is negative on average (mean \(r=-0.042\), positive 0/7). However,
`u` follows `o` in every animal. An environment-specific order interaction is
therefore not identifiable.

## 2026-07-30 — Freeze and qualify the mirror-open continuity control

**Status:** Accepted; design and feasibility gate locked before neural scoring

**Decision:** For every preselected one-step source/wall target, reflect the
target across the source and require that exact seam to be open in the same
held-out target session. Pair relative bins and use identical cells, source
distance, translation axis, strip gap, target session, and square baseline.
Interpret a positive demeaned contrast only as evidence against symmetric
nearby continuity in the supported layout—not as proof of a portable wall
operator.

**Why:** Geometry-only screening yielded 104 label-eligible queries across all
seven animals before neural rates were loaded. The control-specific
paired-support gates retained 72 tangential triplets in five animals. The
global-rate-demeaned
source-to-wall minus source-to-mirror-open advantage is 0.114 and positive in
5/5, but absolute wall transfer and the direct wall-minus-open vector are
positive in 4/5, the raw primary contrast is positive in 3/5, Q74/Q75 lack
support, and the 3-by-3 grid cannot construct the control for normal-axis
translations.

## 2026-07-30 — Reject composition, learning, innateness, and barrier memory

**Status:** Accepted

**Decision:** Keep the failed branches in the lead report and claim boundary.
Do not rescue them through alternative thresholds or post-outcome target
selection.

**Why:** The oriented-wall design has rank 10 for 25 parameters, and the only
two complete local two-wall factorials both use `bit donut`. Their additive
predictor is worse than the best constituent wall (mean difference -0.024,
positive 1/5). The experience difference-in-differences has strict raw-half
coverage in only two animals and fails its stricter sensitivity. A separately
locked neutral-barrier scar primary Pearson-active metric is negative on
average (-0.183), positive in only 1/6 rats, with every leave-one-rat-out mean
negative; the Spearman-active sensitivity is approximately zero (0.001; 3/6
positive). These results provide no evidence for additive scene construction,
learning, innateness, or persistent coordinate-specific barrier memory.

## 2026-07-30 - Promote the limited-transfer result into the active manuscript

**Status:** Accepted

**Decision:** Replace the earlier boundary-normal draft with an active paper
centered on graded reusability: strong exact-location recurrence, weaker but
positive cross-location transfer, and no evidence for a fully portable or
linearly compositional wall primitive. Keep learning, innateness, general
direction selectivity, and a persistent barrier scar outside the claim.

**Why:** This organization makes the novel empirical contrast explicit without
using mathematical language as a substitute for evidence. The manuscript
reports the matched exact benchmark (\(r=0.230\)), shifted prediction
(\(r=0.148\)), behavior-adjusted shifted prediction (\(r=0.119\)), and
shifted-minus-exact penalty (-0.082) together. It also preserves the
same-session mirror-open support limitation and the fixed-order confound. The
compiled 11-page PDF passed a clean TeX build and page-by-page visual audit.

## 2026-08-01 - Fuse boundary and reward results without pooling endpoints

**Status:** Accepted

**Decision:** Replace the wall-only narrative with one manuscript testing a
shared organizing principle in two independent datasets: the direction of a
registered-cell population change can recur when a spatial relation reappears
at another absolute coordinate. Analyse and report the wall and reward
statistics separately. Treat cross-dataset agreement as conceptual
convergence, not as a pooled effect or direct replication of one estimator.

**Why:** The wall analysis contributes an explicit source-to-target spatial
test and geometric controls. The reward analysis contributes an independent
task, an eight-mouse confirmatory endpoint, strong drift/null/parameter tests,
and a nonlinear trial-level behavioral attack. Together they support
structured reuse more broadly than either analysis alone. Their limits differ:
wall prediction remains place-bound and its strictest tangential spatial
control is heterogeneous; reward specificity is strongly shared with speed
and licking and confounds displacement sign with step length. The combined
claim therefore remains below symbolic composition, innateness, portable
amplitude, behavior independence, or a universal map primitive.

## 2026-08-02 - Publish authorized repository updates directly to main

**Status:** Accepted; author workflow preference

**Decision:** For `asim-v/ca1-wall-update-transfer`, push completed, validated,
and explicitly authorized updates directly to `main`. Open a pull request only
when the author requests review or when repository protections require one.

**Why:** This is a single-author scientific release repository, and the author
prefers completed updates to appear immediately rather than receiving a draft
PR to merge separately. Direct pushes still require a clean scope review,
successful relevant checks, a recoverable local commit, and post-push remote
verification.

## 2026-08-02 - Center the contribution on partial transfer

**Status:** Accepted

**Decision:** Describe the wall result as a registered-cell pattern that
partially predicts a distinct seam and loses predictive strength relative to a
query-matched exact seam. Treat the reward arm as transition-class convergence
with substantial behavior-shared geometry. Do not infer an invariant
transformation from either dataset.

**Why:** The central wall estimate is positive in 7/7 animals at a distinct,
non-overlapping seam (mean 0.148), but its exact-location benchmark is 0.230 and
only 1/7 animals favors the shifted predictor. Reward specificity is positive
in 8/8 confirmatory mice, but sign and distance are confounded and speed/licking
attenuate the chosen statistic. The graded claim is more specific than a broad
reusability claim and survives the strongest available controls.

## 2026-08-02 - Accept exhaustive empirical baseline; stop fitted M0--M3 arm

**Status:** Accepted under the frozen protocol

**Decision:** Add the fully enumerated exact-distance empirical spatial rank to
the manuscript and report all companion endpoints. Apply the predeclared stop
to the fitted boundary/place/update comparison and calculate no partial target
endpoint.

**Why:** The correct relation ranks above chance in 7/7 animals and exceeds the
mean alternative in 7/7, so the result is not created by selecting one easy
incorrect orientation. It exceeds the best alternative in only 5/7 and the
strict tangential tier remains heterogeneous. Six animals have only three
exposure cycles and one has two; the combined model's empirical-update weight
cannot be calibrated on neural sessions disjoint from every outer test pair.
Additional independent cycles or a separate development cohort are required.
