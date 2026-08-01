# Partial cross-location generalization in CA1 wall-related remapping

## Verdict

The stronger result is not the discovery of boundary coding—boundary-vector
theory and earlier recordings already anticipate that—nor a fully portable
wall symbol. The data instead support a narrower, empirically testable
organization:

> A cell-resolved wall-minus-open rate-change pattern estimated from other
> environments recurs in a held-out environment and partially generalizes to
> a nearby, non-overlapping wall location. The cross-location prediction is
> consistent across animals but weaker than exact-location prediction.

This is **partial cross-location generalization with strong spatial
dependence**. It constrains what kind of structure is hidden inside CA1
remapping: responses are neither independent across locations nor fully
location invariant.

All results are exploratory reanalyses.  Animals are the biological units;
cells, seams, target queries, and permutations are estimator internals.

## What was predicted

For each registered cell, a local wall edit is the rate beside one oriented
physical seam in a deformed environment minus the rate at the same coordinates
in an independent familiar square.  Training and testing use different
exposure cycles and different square baselines.

The held-out outcome is one local rate residual per registered cell in a later
target session.  Target neural rates never enter a training profile.  The
target wall label, occupancy, and registration support are used to define an
eligible query, so this is local neural-vector prediction rather than blind
prediction of a complete environment.

The key question is whether the **identity of the cells that increase and
decrease** is reusable, not merely whether mean population activity is larger
near a wall.

## Evidence ladder

### 1. Reuse survives an exact local-context match

Training wall and open environments were first restricted to have exactly the
same blocked/accessibility state for every other neighbor of the target
partition.  Only the focal seam state could differ locally.

At the 15 cm target-side strip, the global-rate-demeaned wall-versus-open
predictor advantage remained positive in 7/7 animals:

- 252 eligible held-out local predictions;
- animal-mean advantage: 0.184 Spearman-correlation units;
- horizontal and vertical subsets independently positive in 7/7 animals.

This rules out recurrence of the complete immediate target-partition boundary
configuration as the sole explanation.  Distal global geometry still differs.

### 2. A one-tile geometric counterfactual transfers to other shapes

The sharpest same-location contrast uses training environments `u` and `o`.
Their blocked-partition vectors are identical except that partition 5 is
blocked in `u` and accessible in `o`.  The neural difference learned from this
single-tile edit is evaluated at the same focal wall in held-out target
geometries `i`, `l`, and `t`.

The global-rate-demeaned advantage is positive in 7/7 animals:

- 46 eligible predictions;
- wall-profile mean \(r=0.197\);
- open-profile mean \(r=-0.028\);
- wall-minus-open mean: 0.225.

It is spatially concentrated:

- 2.5--7.5 cm: mean advantage 0.233, positive 7/7 (46 predictions);
- 17.5--22.5 cm: mean advantage 0.069, positive 6/7 (52 predictions);
- descriptive animal-level near-minus-far: mean 0.164, positive 7/7.

The distance bands were scored on separately eligible support—most notably,
Q74 contributes two near predictions and eight far predictions—so this is a
descriptive localization sensitivity, not a query-paired contrast.

Raw-event maps adjusted additively for speed, allocentric movement direction,
and within-session linear time retain the single-tile advantage in 7/7 animals
(mean 0.225 with time; 0.229 without time).  Reconstruction and finite-mask
audits are exact.

This is the cleanest geometric edit in the dataset, but not a randomized
causal contrast: `u` occurs after `o` in all seven animals, with a within-cycle
lag of one to five sessions.  The repeated order and the loss of accessible
area at partition 5 remain inseparable from wall state.

A post-outcome fixed-order falsification rebuilt every query with identical
cells and bins.  A generic later-minus-earlier vector was estimated only from
training pairs whose state at the same seam did not change.  In the complete
all-lag sensitivity:

- actual `u-o` edit versus target: mean \(r=0.195\), positive 7/7;
- generic order vector versus target: mean \(r=-0.042\), positive 0/7;
- actual-minus-order correlation advantage: 0.237, positive 7/7; and
- the vector obtained after subtracting generic order drift from `u-o`
  remained predictive at mean \(r=0.212\), positive 7/7.

An exact-lag version retained 24/46 queries in five animals and was concordant.
This argues against simple generic sequence drift, but it cannot identify an
environment-specific order interaction because `u` never precedes `o`.

### 3. Part of the update transfers to a different physical location

This is a separate, multi-geometry estimator; it does not transport the `u-o`
endpoint difference from section 2. For each held-out target wall, a source
wall-minus-open cell vector was reconstructed at a **different seam exactly one
25 cm grid step away** with the same signed wall normal. Source and target
neural sessions are disjoint. One query-level registered-cell set and spatial
support are imposed on the shifted comparison and its exact-location
benchmark.

For session-global-rate-demeaned maps:

- source effect versus held-out target wall residual:
  mean \(r=0.148\), positive in 7/7 animals;
- source-effect specificity for the target wall residual over the target-open
  profile: mean 0.216, positive in 7/7;
- source wall versus source open as predictors of the target wall residual:
  mean advantage 0.152, positive in 7/7.

The translated source and target strips never share a 5 cm bin. For
one-step translations along the wall tangent their closest bin centers are
5 cm apart; for translations along the wall-normal axis they are 15 cm apart.
Direct reuse of the same spatial samples therefore cannot produce the result.

The translated effect is nevertheless weaker than the exact-location
benchmark:

- query-matched exact-location mean: \(r=0.230\);
- transfer-minus-exact mean: -0.082;
- only 1/7 animal values are positive.

This is why the result is called **partial local transfer**, not a
location-invariant wall code.

### 4. Raw-event behavior adjustment preserves translated prediction

Every cross-location query was reconstructed from raw events using maps
adjusted additively for speed, allocentric movement-direction harmonics, and
within-session linear time. The physical evaluation reference was calibrated
only from the first familiar square.

The eligibility audit matches the stored-map query/session/seam identities and
counts exactly: 405 target queries, 658 source pairs, identical query IDs,
registered-cell counts, source seams, target-bin counts, and source-bin counts.
It does not hash the complete cell-ID or bin-coordinate arrays. With session
time included:

- translated source effect versus held-out target residual:
  mean \(r=0.119\), positive in 7/7 animals;
- specificity relative to the training-derived target-open profile:
  mean 0.172, positive in 7/7; and
- source-wall minus source-open correlation with the target residual:
  mean 0.138, positive in 7/7.

Omitting session time is essentially unchanged (\(r=0.119\), specificity
0.170, and source-wall minus source-open 0.138; all positive in 7/7). This is
an additive nuisance control, not local trajectory matching. Target-session
neural outcomes estimate that session's nuisance coefficients before
evaluation, but never enter source profile training.

### 5. An explicit empirical spatial null is positive but heterogeneous

The first null retains the observed neural maps, registered-cell identities,
held-out target response, and 25 cm seam-midpoint lag, while replacing the
correct source-wall relation with an incorrectly oriented source. The pooled
correct-minus-incorrect advantage is 0.101 and positive in 7/7 animals. Exact
enumeration of all animal-level sign assignments gives an upper-tail fraction
of 1/128, and every leave-one-animal-out mean remains positive.

The result is not uniform across spatial strata. Restricting to tangential
translations with the same 5 cm nearest-strip lag gives an advantage of 0.040,
positive in only 3/6 animals, with a sign-flip tail fraction of 0.25. The pooled
null therefore rules out an arbitrary equal-midpoint match but does not remove
smooth spatial continuity from the strictest distance-matched stratum.

### 6. A mirror-open control narrows the spatial-continuity alternative

Before reading its neural outcome, a stronger control was frozen for every
geometrically eligible source/target relation: reflect the wall target across
the source and require the mirror seam to be open in the **same held-out
session**. The source-to-wall and source-to-open distances are both exactly
25 cm, their nearest strip bins are both 5 cm away, and the evaluation uses
paired relative bins, the same cells, target session, and square baseline.

The 3-by-3 grid permits an exact internal mirror only for translations along
the wall tangent. After the control-specific paired wall/open support gates,
72 triplets remain in five animals; Q74 and Q75 have no supported triplets.

- Global-rate-demeaned source-to-wall minus source-to-mirror-open:
  mean 0.114, positive in 5/5 covered animals.
- Absolute source-to-wall correlation:
  mean \(r=0.068\), positive in 4/5.
- Source correlation with the direct within-session wall-minus-open vector:
  mean \(r=0.077\), positive in 4/5.
- The raw-rate primary contrast is positive in only 3/5.
- The exact animal-level sign-flip fraction for the demeaned advantage is
  1/32, and all leave-one-animal-out means are positive.

Thus purely symmetric nearby population continuity is disfavored for the
covered, demeaned tangential subset. It is not eliminated for the full result:
absolute transfer is weak in two covered animals, two animals lack support,
and no exact mirror control is possible for normal-axis translations.

### 7. Registered-cell alignment matters, but not uniformly

A 999-draw identity diagnostic uses one coherent global cell-ID mapping per
animal and draw, shared across all target queries and rate modes while
preserving every query-specific common-cell set.

For the global-rate-demeaned translated effect, the one-sided descriptive
tails are 0.007, 0.001, 0.001, 0.069, 0.005, 0.002, and 0.060.  Five animals
show strong separation from the restricted identity null; Q51 and Q75 do not
cross a conventional 0.05 threshold.  Only subsets of cells are exchangeable,
so these numbers are identity diagnostics rather than biological
population-level \(p\)-values.

## What the data do not support

### General direction selectivity

The pooled same-signed-wall-normal versus different-signed-wall-normal
comparison was initially positive in 7/7 animals, but exact midpoint distance
did not equate the spatial layouts of the controls. Some opposite-facing strips
overlap by five bins,
whereas opposite-away strips are separated by at least 30 cm.

After stratifying those layouts:

- same versus opposite signed wall normal for translations along the wall
  tangent is positive in only 3/6 comparable animals;
- same-signed-normal transfer along the wall-normal axis versus
  opposite-facing/overlapping transfer is positive in 4/7;
- same-signed-normal transfer versus spatially separated opposite-away
  transfer is positive in 7/7, but changes both layout and distance.

Thus a broad “direction-selective transfer” claim is not identified. The
defensible statement is only that the preselected same-signed-normal, one-step
source effect transfers and is specific relative to a training-derived
target-open profile control.

### Additive composition

The ten unique geometries do not identify a general held-out-shape additive
wall model: an oriented-wall design has rank 10 for 25 parameters.  Only two
complete local two-wall factorials exist, both using the same `bit donut`
target.

In the leakage-safe, global-rate-demeaned test, an additive two-wall predictor
was worse than the better constituent wall:

- additive mean \(r=0.153\);
- oracle best-single mean \(r=0.178\);
- additive-minus-best-single: -0.024, positive in only 1/5 animals;
- additive normalized error was worse in 5/5.

No portable additive wall grammar is supported.

### Learning, innateness, or a persistent barrier memory

The repeated QLAK cycles are perfectly confounded with elapsed time and
handling.  A wall-versus-open recurrence difference-in-differences is
directionally promising in stored maps, but strict raw-half coverage collapses
to two eligible animals and the stricter sensitivity leaves one negative
animal.

An independently locked test in all six neutral-barrier rats from Blair et al.
also failed. The primary Pearson-active reliability-adjusted pre/post change
at former-barrier bins 11--13 minus symmetric route controls was positive in
only 1/6 rats, with animal mean -0.183 and every leave-one-rat-out mean
negative. The Spearman-active sensitivity was approximately zero (mean 0.001;
3/6 positive). The result does not support a persistent coordinate-specific
barrier “scar.”

The present analyses therefore say nothing about innateness and do not
identify learning.

## What is—and is not—new

A generic “mental map” does not specify whether environments use unrelated
conjunctive codes, reusable local changes, or portable additive primitives.
The present result distinguishes those broad possibilities: reuse is
measurable, including partial transfer across coordinates, but it loses
strength and fails the available additive test.

That is not the most demanding novelty comparison, however. Fixed-input
boundary-vector models already predict cell-specific place-field changes in
novel geometries, Rivard et al. showed identified CA1 cells following a moved
barrier, and Lee et al. demonstrated registered-cell cross-environment spatial
correspondence in these data. Partial cross-location boundary sensitivity is
therefore anticipated.

The additional contribution is statistical and quantitative: isolate a
subtractive registered-cell contrast, exclude target neural rates from the
training profiles, transport the contrast to non-overlapping coordinates, and
measure both its surviving information and its loss relative to the exact
location. The result constrains boundary-conditioned remapping; it does not
discover boundary coding.

## Claim ceiling

The strongest defensible sentence is:

> In an exploratory longitudinal CA1 reanalysis, a registered-cell
> wall-minus-open contrast learned from training geometries carried
> out-of-sample information about a later wall response at a preselected,
> non-overlapping neighboring seam; transfer was weaker than exact-location
> reuse and did not provide evidence for linear two-wall composition.

Do not replace this with:

- “CA1 contains a portable wall primitive”;
- “CA1 transfer is generally direction selective”;
- “CA1 composes maps from walls”;
- “experience learns the wall code”; or
- “the result demonstrates an innate mental map.”

## Relation to prior work

Lee et al. showed that local boundary distance and direction predict the
cross-environment representational structure of these CA1 data
([paper](https://doi.org/10.1016/j.neuron.2024.10.027);
[data](https://doi.org/10.5281/zenodo.13993254)).  Rivard et al. previously
identified individual barrier-attached CA1 cells whose fields followed a moved
barrier ([paper](https://doi.org/10.1085/jgp.200409015)), and fixed-input
boundary-vector models already predict systematic cell-specific field
transformations under boundary changes
([Hartley et al.](https://pubmed.ncbi.nlm.nih.gov/10985276/)).

The present contribution is narrower and incremental relative to the general
boundary-vector idea:
it tests the registered-cell **change vector**, excludes target neural rates
from training, transports that vector between non-overlapping physical seams,
quantifies the exact-location loss, and directly records the failure of
additive two-wall composition.

## Decisive follow-up experiment

The most dangerous remaining explanation is a smooth local deformation
response: broad neighboring place fields, lost accessible area, or shared
turning/edge-following policy could correlate nearby source and target
contrasts without a translated wall component. Positive transfer in some
opposite-direction layouts and failed direction controls make this a serious
alternative. The mirror-open audit argues against a purely symmetric version
of this account for the supported tangential subset, but cannot test the
normal-axis subset or remove behavior and whole-geometry confounds.

The current dataset cannot separate wall appearance, blocked transitions,
lost accessible area, navigation policy, spatial autocorrelation, and repeated
order. A decisive experiment should independently randomize:

1. wall position and signed normal;
2. a traversable visual stripe versus a thin physical barrier;
3. the same barrier with an open versus closed door;
4. insertion, removal, and session order; and
5. single walls versus factorial two-wall combinations; and
6. source-target distance, including targets beyond ordinary place-field
   spatial autocorrelation.

The primary endpoint should be prediction of held-out registered-cell update
vectors at new positions and orientations, with matching-orientation transfer
required to exceed equidistant wrong-orientation, open-door, sham, and locally
kinematic-matched controls. That design would test whether the partial transfer
seen here is genuinely boundary-derived, transition-derived,
behavior-derived, or simply smooth spatial continuity.

## Tracked evidence

- `results/source_data/boundary_fragment_context_matched.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_near.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_far.json`
- `results/source_data/boundary_fragment_single_tile_counterfactual_behavior_adjusted.json`
- `results/source_data/boundary_fragment_single_tile_order_placebo.json`
- `results/source_data/boundary_fragment_cross_location_transfer.json`
- `results/source_data/boundary_fragment_cross_location_behavior_adjusted.json`
- `results/source_data/boundary_fragment_cross_location_behavior_adjusted_eligibility_audit.json`
- `results/source_data/boundary_fragment_cross_location_mirror_open.json`
- `results/source_data/boundary_fragment_cross_location_spatial_controls.json`
- `results/source_data/boundary_fragment_cross_location_cell_permutation.json`
- `results/source_data/boundary_compositional_counterfactual.json`
- `results/source_data/boundary_fragment_experience_did.json`
- `results/source_data/boundary_fragment_experience_raw_split.json`
- `results/source_data/blair_barrier_scar.json`
- `results/source_data/local_wall_update_transfer_figure.csv`
- `reports/figures/local_wall_update_transfer.png`
- `data/metadata/zenodo_13993254_manifest.json`

Focused methods and limitations are in:

- `reports/boundary_compositional_counterfactual_audit.md`;
- `reports/boundary_fragment_cross_location_audit.md`;
- `reports/boundary_fragment_cross_location_behavior_adjusted.md`;
- `reports/boundary_fragment_cross_location_mirror_open_audit.md`;
- `reports/boundary_fragment_single_tile_order_placebo_audit.md`;
- `reports/experience_identifiability_audit.md`;
- `reports/blair_barrier_scar_design_lock.md`; and
- `reports/blair_barrier_scar_results.md`.

Regeneration commands are listed in `results/source_data/README.md`. Each JSON
artifact records its design and analysis settings; the behavior-adjusted
artifact additionally hashes the estimator, scorer, and adapter used for its
validated per-animal cache. Raw public inputs and large operational
checkpoints remain ignored. The Zenodo manifest records the primary input
sizes and published MD5 values; the Blair artifact records its release source
and schema audit. The release snapshot is tagged
`milestone-local-wall-update-transfer-2026-07-30`.
