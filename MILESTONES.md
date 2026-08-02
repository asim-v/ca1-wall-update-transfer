# Milestones

This file is the authoritative project tracker. Update checkboxes only when the
corresponding evidence exists, and record every gate decision in
`DECISIONS.md`.

## M0 — Scope and feasibility

- [x] Define one-sentence scientific question.
- [x] Identify primary public dataset and publication.
- [x] Define the primary geometric object and estimand.
- [x] Record major confounds and failure criteria.
- [x] Verify data license, file structure, and required metadata.
- [x] Verify that cell identity and geometry labels support within-animal
      comparisons.
- [x] Produce a data dictionary for one animal.

**Exit gate:** Required signals and metadata are accessible without reconstructing
the original imaging pipeline.

## M1 — Reproduction

- [x] Download one animal only.
- [x] Load behavior, neural activity, geometry, and session labels.
- [x] Reproduce one published occupancy map.
- [x] Reproduce one published CA1 rate-map or similarity result.
- [x] Save a deterministic environment and dependency lock.

**Exit gate:** One known result is reproduced closely enough to establish data
and preprocessing validity.

## M2 — Synthetic geometry validation

- [x] Implement flat, isotropic population-code simulation.
- [x] Implement known normal-anisotropy simulation.
- [x] Replay simulations along real animal trajectories and arena masks.
- [x] Validate Jacobian and metric calculations.
- [x] Confirm basic sign recovery and an unequal-occupancy null.
- [x] Add initial automated regression tests.
- [ ] Confirm recovery across the frozen bandwidth and neuron-count
      ranges.

**Exit gate:** The estimator recovers the known sign and does not invent
boundary-normal magnification in the flat null.

## M3 — One-animal biological pilot

- [x] Select one animal and one high-coverage boundary manipulation without
      inspecting the target outcome.
- [x] Fit cross-validated, boundary-aware population maps.
- [x] Estimate local metric tensors.
- [x] Measure split-half reliability.
- [x] Compute \(A(d)\) against the frozen square-session drift comparator.
- [x] Run square-square and coarse occupancy-match controls.
- [x] Test the frozen bandwidth range.
- [x] Run trace-position and spatial-orientation controls after the estimator
      reliability gate passes.

**Exit gate:** Metric reliability exceeds null and the anisotropy sign is stable
across reasonable analysis choices.

**Current gate result:** Animal 51 failed and remains development-only. After
the fold redesign, the eligible `+` cohort produced a positive primary contrast
exceeding the frozen square-session drift comparator in all 12 sequences, but
only 3/12 sequences passed the complete reliability gate. The comparator used
square-pair-valid rather than target-valid support. The directional aggregate
is promising; the frozen confirmatory criterion is not fully satisfied.

## M4 — Behaviorally eligible cohort

- [x] Freeze the pilot analysis configuration.
- [x] Run all eligible mice for the fixed `+` condition.
- [x] Perform common-cell and equal-neuron-count analyses.
- [ ] Fit mouse/session-level hierarchical or permutation inference.
- [ ] Estimate effect amplitude and decay length with uncertainty.
- [ ] Compare deformations and square reinstatement without interpreting the
      latter as isolated boundary removal.
- [ ] Complete all primary robustness checks.

**Exit gate:** The effect generalizes across biological replicates or a
well-powered null result is established.

## M5 — Learning and mechanism

- [x] Quantify reliability across repeated geometry sequences.
- [x] Separate changing effect magnitude from increasing stability.
- [ ] Apply the same estimand to candidate cognitive-map models.
- [ ] Determine whether boundary-vector models uniquely predict the observed
      directional signature.
- [ ] Decide whether a second public dataset is necessary.

**Exit gate:** The result has a defensible learning or mechanistic
interpretation beyond descriptive remapping.

**Current gate result:** Common-support effect magnitude increased in two
animals and decreased in two; two of four reliability trajectories were
nonmonotonic. This dataset analysis does not support the
experience-stabilization claim, and candidate-model mechanism tests remain
open.

## M6 — Manuscript-ready package

- [x] Re-run novelty search and update related work.
- [x] Freeze inclusion criteria and analysis version.
- [x] Produce the three primary figures and source-data tables.
- [x] Complete reproducibility documentation.
- [x] Run the full test suite from a clean environment.
- [x] Write limitations and falsification results.
- [x] Archive code and derived, redistributable artifacts.

**Exit gate:** A reader can reproduce every reported result from documented
inputs and commands.

**Current gate result:** A complete working manuscript, DOI-checked
bibliography, deterministic figure generator, tracked manuscript figures, and
visually verified 15-page PDF exist. That manuscript snapshot passed 35 tests;
the later exact-location package installed from the lock in a fresh
environment and passed 86 tests. The current stronger-claim working tree
passes 147 tests, but a fresh raw-artifact recomputation of this latest package
remains an M8 follow-up rather than an uncompleted M6 gate.

## M7 — Discovery pivot: exact-location wall-conditioned cell profiles

- [x] Suspend manuscript-first work and open a falsification-led discovery
      ledger.
- [x] Audit raw trajectory, released-map, and partition-row coordinates with
      asymmetric arena shapes.
- [x] Test exact oriented-seam wall/wall versus wall/open edit repeatability
      within fixed pairs of global geometries.
- [x] Verify animal-level direction, horizontal/vertical subsets, cell-ID
      dependence, and near-versus-far localization.
- [x] Replace within-record label shuffling with a restricted,
      session-consistent, common-cell-set-preserving global-ID permutation
      diagnostic.
- [x] Implement target-rate-held-out local prediction across non-overlapping
      exposure cycles and square baselines, with target-aware labels and
      support disclosed.
- [x] Verify the symmetric open-target reversal and alternative equal
      exposure/environment/seam weighting.
- [x] Remove session-wide cell-rate and occupancy explanations.
- [x] Reconstruct the result from non-overlapping raw-event halves.
- [x] Adjust raw-event rate estimates for speed and allocentric movement
      direction near the seam.
- [ ] Decide whether experience strengthens wall-specific reuse beyond
      open/open and general map reliability.
- [x] Freeze a compact figure/source-data package.
- [x] Verify the locked dependencies install in a fresh environment and all
      tests pass.
- [ ] Recompute every raw-event artifact from scratch in that fresh
      environment.

**Exit gate:** The exact-location wall profile beats exact-location open and
same-signed-normal different-wall controls in animal-level summaries, remains
localized near the wall, preserves registered-cell identity, and survives
non-overlapping raw-event and kinematic controls.

**Current gate result:** Stored-map and cross-exposure local prediction are positive
in 7/7 mice. The global-rate-demeaned held-out predictor advantage has animal
mean 0.214. The restricted, common-cell-set-preserving global-ID diagnostic
has global-rate-demeaned one-sided tails at most 0.005 in every mouse (raw Q51
is a non-pass), and far-wall strips are near zero.
Non-overlapping raw-event halves retain the advantage in 14/14 coverage-eligible
exposure blocks and 5/5 eligible mice; the stricter dwell sensitivity retains
10/10 blocks in 4/4 mice. Common-reference additive adjustment for speed,
allocentric movement direction, and session time remains positive in 7/7
mice (animal mean 0.209). See `DISCOVERY.md` and
`reports/boundary_component_discovery.md`.

**Static-result decision:** Pass. Experience-dependent stabilization remains
outside the claim until the open/open reliability comparator or a prospective
experiment succeeds.

## M8 — Stronger-claim audit: local update reuse and transfer

- [x] Hold every non-focal neighbor state of the target partition fixed and
      retest the focal wall/open prediction.
- [x] Isolate the exact `u` versus `o` single-tile geometric counterfactual.
- [x] Re-estimate the single-tile result from raw events after additive speed,
      allocentric movement-direction, and session-time adjustment.
- [x] Test generic same-seam later-minus-earlier drift under the fixed
      environment order.
- [x] Transport the learned wall-minus-open registered-cell vector to a
      non-overlapping seam one 25 cm grid step away.
- [x] Audit spatial overlap, transport loss, target wall-state specificity,
      signed-direction controls, and coherent cell-identity permutations.
- [x] Test the available leakage-safe two-wall additive counterfactual.
- [x] Test experience with an open/open reliability comparator and raw
      non-overlapping halves.
- [x] Lock and execute an independent neutral-barrier scar prediction.
- [x] Freeze a stronger-claim report, figure, and source-data package.
- [x] Complete the raw-event kinematic adjustment of the cross-location
      transport result in all seven animals.
- [x] Freeze and run a same-session, mirror-lag open-seam control for generic
      nearby spatial continuity.

**Exit gate:** The focal wall update survives exact local-context matching and
the one-tile counterfactual, predicts a held-out cell vector at a
non-overlapping location with quantified transport loss, and is stated without
claiming direction invariance, composition, learning, or causality unless
their dedicated controls pass.

**Current gate result:** Pass as an exploratory internal result. Exact
local-context matching is positive
in 7/7 animals (mean advantage 0.184), and the single-tile counterfactual is
positive in 7/7 (mean 0.225) before and after raw-event kinematic adjustment.
The separately reconstructed multi-geometry source vector predicts the
held-out one-step target residual in 7/7 animals (mean \(r=0.148\)) and is
specific relative to a training-derived target-open profile, but it is weaker
than its query-matched exact-location benchmark (\(r=0.230\)) by 0.082.
The raw-event kinematic reconstruction retains transfer at mean \(r=0.119\),
positive in 7/7, with identical 405 target queries and 658 source pairs.
For the geometrically covered tangential subset, a frozen same-session
mirror-open control favors the wall target after global-rate demeaning by
0.114 in 5/5 covered animals, but absolute wall transfer and raw-rate signs are
not uniform and Q74/Q75 lack support.
General direction selectivity fails layout-matched controls, identity support
is nonuniform, the additive two-wall model loses to the best individual wall,
the learning extension is underidentified, and the independent barrier-scar
prediction fails.

## M9 - Limited-transfer working manuscript

- [x] Replace the earlier boundary-normal draft with the active
      limited-transfer paper.
- [x] Separate exact-location reuse, cross-location transfer, and composition
      as distinct estimands.
- [x] Integrate behavior, mirror-open, identity, direction, order, experience,
      and independent-dataset controls without promoting failed branches.
- [x] Add a significance statement, evidence-ladder table, formal methods,
      prospective design, data availability, and DOI-checked bibliography.
- [x] Compile the LaTeX source without document warnings.
- [x] Render and visually inspect every page of the final PDF.
- [x] Create a tagged, verified full-history bundle backup.
- [x] Add author identity, institutional affiliation, and contact information.
- [x] Publish a curated public repository containing the manuscript, derived
      source data, code, tests, audits, and data-provenance boundary.

**Exit gate:** The active manuscript states a novel, testable contribution
without describing partial transfer as a location-invariant primitive,
composition, learning, or innateness, and the final PDF passes a page-by-page
layout audit.

**Current gate result:** Pass. The 8-page, two-column working manuscript is
`manuscript/main.tex`; the stable compiled deliverable is
`output/pdf/ca1_wall_update_transfer_manuscript.pdf`. The primary claim is that
a registered-cell wall-related update transfers across a non-overlapping
25 cm displacement but loses 0.082 correlation relative to exact-location
reuse. All principal limitations and failed claims remain visible.

## Current next actions

1. Reset the arXiv account password, sign in, and complete the author-only
   declarations and license choice.
2. Upload the locally validated source package and audit the server-generated
   PDF before the final submission action.
3. Complete funding disclosures and confirm the CRediT statement.
4. Design a counterbalanced experiment that independently manipulates wall
   position, traversability, lost area, and single-versus-paired walls.
5. Place target barriers beyond ordinary place-field autocorrelation and
   preregister remote transfer against equidistant open and wrong-orientation
   controls.

## M10 - arXiv submission

- [x] Create a minimal, self-contained LaTeX source package.
- [x] Include the top-level TeX file, bibliography, generated BBL, and required
      figure using arXiv-safe file names.
- [x] Compile the exact ZIP contents without document warnings.
- [x] Render and visually inspect every page of the arXiv-targeted PDF.
- [x] Prepare title, abstract, author, affiliation, category, comments, and
      repository metadata.
- [x] Complete the author account registration and email verification.
- [ ] Confirm author declarations and the arXiv distribution license.
- [ ] Upload the source package and pass arXiv AutoTeX.
- [ ] Inspect the arXiv-generated PDF and metadata preview.
- [ ] Complete the final Submit Article action.

**Exit gate:** The server-generated PDF and metadata match the locally audited
submission, the author has confirmed the personal and legal declarations, and
arXiv has accepted the final submission into its moderation queue.

**Current gate result:** The 657,265-byte source ZIP at
`submission/ca1_wall_update_transfer_arxiv.zip` contains five required entries
and recompiles cleanly after extraction. The package SHA-256 is
`254906876768A8D2F243D9DE99A0CFA892CE63D8F8D900C404EB01958E54618F`.
The `neuroemi` account and institutional-email verification are complete.
Upload remains blocked on resetting or recovering the author-chosen password,
then signing in to make the personal declarations and license choice.

## M11 - Partial-generalization revision

- [x] Enumerate exact animal-level sign assignments for the empirical spatial
      controls and record leave-one-animal-out stability.
- [x] Separate the broad equal-midpoint wrong-orientation result from the
      strictly distance- and strip-matched tangential sensitivity.
- [x] Lead the title, abstract, significance statement, results, and discussion
      with the supported phenomenon rather than unsupported mechanisms.
- [x] Move composition, direction, experience, and former-barrier endpoints to
      a compact supplementary exploratory section.
- [x] Regenerate the primary figure, manuscript PDF, metadata, and arXiv source
      package.
- [x] Compile the exact extracted ZIP and visually audit all 8 pages.
- [x] Publish the revised manuscript, null calibration, source artifact, code,
      and regression test to the public reproducibility repository.
- [x] Convert the manuscript to a compact two-column layout, keep all three
      figures and the evidence table at full width, and balance the final
      references.
- [x] Remove editorial stage directions and replace specialist shorthand with
      direct descriptions of the measured comparisons.
- [x] Add a data-backed explanatory figure that pairs exact-location with
      cross-location prediction and places both spatial controls beside it.

**Exit gate:** The manuscript makes a firm empirical claim of partial,
spatially dependent cross-location generalization while retaining the fixed-
order and smooth-spatial-continuity limitations needed to constrain mechanism.

**Current gate result:** Pass. Correctly oriented sources outperform the pooled
equal-midpoint wrong-orientation control by 0.101 in 7/7 animals (descriptive
one-sided sign-flip fraction 1/128), and all leave-one-animal-out means remain
positive. The strict tangential 25 cm midpoint and 5 cm strip-matched contrast
is only positive in 3/6 animals (fraction 0.25), so the paper does not claim a
general orientation-specific operator. The revised 8-page paper instead
states the supported result: cross-location prediction is reliable but weaker
than exact-location reuse. Its third figure makes that gradient and the
boundary of the spatial controls visible directly.

## M12 - Fused boundary-and-reward manuscript

- [x] Reframe the common question as reuse of a registered-cell population-
      change direction across absolute locations.
- [x] Keep the wall and reward estimators separate and prohibit numerical
      pooling across datasets.
- [x] Integrate the eight-mouse confirmatory reward endpoint, deconvolved-event
      replication, drift controls, trial and transition-label nulls, parameter
      sensitivities, and registration audits.
- [x] Integrate the corrected behavior-coordinate audit and the frozen
      nonlinear out-of-fold trial model.
- [x] Rewrite the title, abstract, significance statement, introduction,
      results, discussion, dataset description, methods, evidence table, and
      data-availability statement around the combined evidence.
- [x] Add the reward-control and trial-behavior figures to the two-column paper.
- [x] Compile and visually inspect all 13 pages, including five figures, the
      full-width evidence table, equations, references, and final-column
      balance.
- [x] Build a minimal seven-file arXiv source archive while preserving the
      prior wall-only archive as a backup.

**Exit gate:** The fused paper must gain generality from two independent
paradigms without presenting either result as a universal, coordinate-free, or
behavior-independent hippocampal operator.

**Current gate result:** Pass locally. The boundary arm shows cross-location
prediction in 7/7 mice with a lower value than exact-location reuse. The reward
arm shows signed-displacement specificity in 8/8 confirmatory mice; nonlinear
trial-level speed and lick conditioning leaves dF/F specificity in 7/8 while
reducing the cohort mean to 49.5% of the original. The audited PDF is
`output/pdf/ca1_reusable_population_updates_manuscript.pdf`. The arXiv ZIP is
1,141,988 bytes with SHA-256
`BA49D3B2771F744F8B8C492D57276E5D616F8FECAF6BCE9417B2E2432F6FC59E`.

## M13 - Public fused release

- [x] Publish the reward-analysis companion repository without raw recordings.
- [x] Replace the manuscript's provisional availability language with the
      permanent companion URL.
- [x] Recompile and visually inspect the affected manuscript page.
- [x] Rebuild the seven-file arXiv ZIP with its `figures/` directory preserved.
- [x] Compile the exact extracted ZIP successfully.
- [x] Prepare the fused manuscript, figures, metadata, and release artifacts
      for the public boundary repository.
- [x] Merge public boundary-repository PR #1 into `main`.

**Exit gate:** Both analysis repositories are public, cross-linked, and contain
the code, derived endpoints, provenance, and audited manuscript artifacts
needed to inspect the fused claim without redistributing source recordings.

**Current gate result:** Pass. The reward companion is public at
`https://github.com/asim-v/ca1-goal-update-transfer`. Boundary PR #1 was merged
into `main` as commit `c9a5d794293acfdd76c99d704ee98265c99d9cb4`. The updated
manuscript is 13 pages; its PDF SHA-256 is
`CB01CEE1BB9077A5F28A862405FE1F7D02CB10A768A71613CDE496E5240AE197`.
The validated arXiv ZIP is 1,141,988 bytes with SHA-256
`BA49D3B2771F744F8B8C492D57276E5D616F8FECAF6BCE9417B2E2432F6FC59E`.

## M14 - Remove Significance statement

- [x] Remove the standalone Significance statement from the working manuscript.
- [x] Remove the same section from the arXiv source.
- [x] Recompile the manuscript and confirm the front matter flows directly from
      keywords to the Introduction.
- [x] Render and visually inspect all 12 resulting pages.
- [x] Rebuild the seven-file arXiv ZIP with the `figures/` directory preserved.
- [x] Compile the exact extracted arXiv ZIP and confirm the section is absent.
- [x] Prepare the updated PDF, metadata, hashes, and public-repository branch.
- [x] Publish the public repository update directly to `main`.

**Exit gate:** No active manuscript or arXiv source contains a Significance
statement, and the shortened document retains a clean two-column layout.

**Current gate result:** Pass. Commit
`4606501b03dcd200814c6c340bea495ccde32ab7` was pushed directly to public
`main`; GitHub consequently marked PR #2 merged. The manuscript is now 12
pages. Its PDF SHA-256 is
`7152BB184859F782B829F87C917D700B5B1C4FE8F220F1B5372A62BD18FFB7B4`.
The validated arXiv ZIP is 1,141,803 bytes with SHA-256
`12260E50DB54CF36C3AAD054681EF7CCA6053AD4A7134A33EA55494FF2F649A1`.

## M15 - Specificity and positive-model adjudication

- [x] Audit repository provenance, manuscript numbers, claim levels, current
      primary literature, and reviewer attacks without running a new endpoint.
- [x] Commit the positive-model protocol before inspecting a new neural target
      endpoint.
- [x] Implement and test the fully enumerated exact-distance empirical spatial
      baseline.
- [x] Record every frozen-local outcome, including the best-alternative failure
      and support-limited tangential tier.
- [x] Apply the frozen stop to the fitted M0--M3 boundary comparison rather than
      relaxing its session-leakage barrier.
- [x] Recenter the title, abstract, introduction, results, figures, and
      discussion on partial transfer and its exact-location decrement.
- [x] Add sentence-level machine traceability for every quantitative abstract
      claim and record the absence of a Significance statement.
- [x] Rebuild and visually verify the active PDF and extracted arXiv source.
- [x] Commit the final manuscript revision and publish the validated release
      directly to public `main`.

**Exit gate:** The paper must distinguish descriptive registered-cell transfer
from model-level residual information and from an invariant transformation. It
must report the exhaustive empirical baseline without hiding the strongest-
alternative and tangential failures, and every abstract number must resolve to
a tracked result.

**Current gate result:** Pass. Manuscript commit
`565e1b2e415fc1caca7b56843100d2cdd4e74241` was released directly to public
`main` in commit `03eaaf71937a8683e9ec6be29817e0cb42441b63`.
The correct wall relation has mean percentile rank 0.612 among all admissible
exact-distance sources and is above chance in 7/7 animals. It exceeds the mean
alternative in 7/7 but the best alternative in only 5/7; the tangential tier is
positive in 4/6. The fitted comparison is not identifiable leakage-free with
the available exposure cycles. The revised claim remains descriptive partial
transfer, with a smaller reward relation after behavioral conditioning.
The public clone passes 156 tests with one raw-data-dependent test skipped, and
all 20 quantitative abstract claims resolve to pinned machine-readable files.
