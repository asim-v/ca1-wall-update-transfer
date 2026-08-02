# Research Log

## 2026-07-28

- Completed a literature and public-dataset scan through 2026-07-28.
- Selected the CA1 boundary-normal metric project.
- Identified two principal backups:
  1. Direct Fisher-Rao audit of the published CA1 hyperbolicity claim.
  2. Cue-locked population-trajectory geometry during flexible navigation.
- Rejected a generic geometry analysis of the longitudinal orthogonalized
  state-machine dataset because a 2026 preprint already applies a
  geometry-aware latent model to that dataset.
- Initialized the research repository structure and milestone gates.

## 2026-07-29

- Verified the paper's 31-session structure, shared square endpoints, cell
  registration, and 5 cm spatial grid.
- Verified Zenodo v1 as CC-BY-4.0 and recorded all file checksums.
- Located the official MIT-licensed analysis code:
  https://github.com/jquinnlee/georepca1.
- Identified a published-method discrepancy: stated Gaussian smoothing is
  5 cm in the paper, 2.5 cm in the repository README, and 7.5 cm in the
  current code default. Raw position and event traces will therefore be used
  for a frozen bandwidth analysis.
- Narrowed the novelty claim to within-condition local metric allocation rather
  than directional remapping or displacement.
- Implemented local-linear Jacobian estimation, cross-fold metric tensors,
  strip-aggregated boundary anisotropy, shared response scaling, synthetic
  codes, and six initial regression tests.
- Recovered the known synthetic boundary-normal sign while an unequal-occupancy
  stationary-code null remained near zero.
- Downloaded and MD5-verified QLAK-CA1-51 (296,158,866 bytes) and generated a
  21-session schema/data dictionary.
- Reconstructed the released sampling, unsmoothed, and smoothed maps exactly;
  documented the fixed-bin, event-probability, dwell-second, and 16 × 16
  pre-crop conventions.
- Replayed a known normal warp and flat null on the animal's measured
  trajectories and mask. The correct sign was recovered at every supported
  bandwidth and the flat-null anisotropy remained much smaller.
- Ran the frozen center-occlusion biological pilot. It failed the reliability
  gate because of sparse foldwise support, weak split-pair correlations,
  fold/bandwidth sensitivity, and square-square changes of comparable size.
  QLAK-CA1-51 was designated estimator-development data.
- Audited novelty against Wang et al. (2020) and Tanni et al. (2022), withdrew
  the broad first-Jacobian/first-boundary-normal claim, and retained only the
  longitudinal crossvalidated true-wall-versus-pseudo-wall residual.
- Implemented guarded, position-balanced temporal-block folds for validation
  on synthetic trajectory replay before applying them to an untouched animal.
- Audited all repeated deformations in QLAK-CA1-08 using positions only. The
  `+` condition had the strongest query support and was carried forward as an
  explicitly exploratory, behavior-selected follow-up.
- Found and corrected a runner mismatch: eligibility used query-design-balanced
  folds while the neural runner still used the older coarse-occupancy folds.
  The provisional animal-08 and animal-56 outputs were invalidated and rerun;
  the correction and its post-outcome timing are recorded in `DECISIONS.md`.
- Downloaded all seven animal archives individually and verified every file
  against its published MD5. The complete positions-only `+` screen retained
  animals 08, 30, 50, and 56 and excluded animals 74 and 75; animal 51 remains
  estimator-development data.
- Committed the frozen estimator and complete eligibility table at `443d038`
  before loading neural traces from animals 30 or 50.
- Ran every eligible `+` sequence. At the primary 10 cm bandwidth, all 12
  estimates had positive normal magnification, positive normal-minus-tangent
  contrast, and a target contrast larger than the absolute frozen
  square-session drift comparator. Frozen sequence gates passed 0/3 in animal
  08, 0/3 in animal 30,
  1/3 in animal 50, and 2/3 in animal 56; every failed primary sequence failed
  the fold-pair reliability threshold.
- The cross-animal directional sign is promising, but the complete frozen
  confirmatory criterion and the experience-dependent stabilization claim are
  not yet met. Common-query longitudinal reliability and falsification
  controls are in progress.
- Restricted all repetitions to identical within-animal query intersections.
  All 12 contrasts remained positive, while exposure-3-minus-1 effect size was
  negative in two animals and positive in two; stabilization was not supported.
- Enumerated all 256 boundary-segment normal/tangent label assignments. The
  observed cohort animal mean had a two-sided calibration fraction of
  0.0078125, explicitly not a randomized-experiment p-value.
- Ran 19 population-shared circular trace-position surrogates in each clean
  prospectively locked animal. Every observed exposure and animal mean
  exceeded every
  surrogate; the plus-one p-value was 0.05, the staged run's minimum
  resolution.
- Re-estimated local spatial slopes after joint adjustment for speed and first
  and second allocentric movement-direction harmonics. All 12 primary
  contrasts and normal components
  stayed positive with identical query support; the four animal means changed
  only from 0.606/0.529/2.005/0.770 to
  0.657/0.540/2.115/0.718.
- Recorded the milestone verdict: retain the consistent descriptive
  directional observation; the analysis does not support full confirmatory
  reliability or experience-dependent stabilization.
- Drafted a complete LaTeX reanalysis manuscript with the split verdict in the
  title, abstract, results, discussion, and methods: 12/12 positive aggregate
  directional contrasts, but only 3/12 complete frozen gates and no consistent
  experience-dependent stabilization.
- Added a DOI-checked bibliography, three primary figures, one supplementary
  control figure, a deterministic figure generator using tracked source JSON,
  and complete build instructions.
- Compiled the paper with Tectonic 0.16.9. The final 15-page PDF had no LaTeX,
  BibTeX, citation, cross-reference, or box warnings. All pages were rendered
  and visually inspected; the repository test suite passed 35 tests.
- A final implementation audit found that the frozen square-session comparator
  used square-pair-valid support while the target used stricter triplet-valid
  support. The manuscript now reports both near-query counts, treats the
  comparator as drift across the complete nine-deformation sequence rather
  than a matched sham, and leaves a same-mask sensitivity as explicit future
  work.

## 2026-07-30

- Suspended manuscript-first work and opened a falsification-led discovery
  campaign using the released longitudinal cells, raw positions, and event
  traces.
- Screened and rejected first blocked-transition decoding, local geometric
  hysteresis, exact boundary aliases, successor/predecessor anticipation,
  blocked-field relocation and learned suppression, imaging-plane
  microtopography, an east-cue reliability gradient, special first-encounter
  recruitment, and an acute impossible-transition population response.
- Found that the alias full-map trend depended on unreliable maps and a sparse
  outlier; exact displacement controls reproduced the apparent neural
  sharpening. A direct decoder showed only the source paper's general
  familiarity-related improvement.
- Found that a matched later wall visit predicted the late-session local map
  better than the first qualifying visit in 6/7 mice, ruling out special
  first-encounter imprinting.
- Found that raw activity at aborted wall approaches was positive relative to
  open crossings in only 4/7 mice. Residual effects occurred after reversal
  and could not be separated from turning/deceleration because matched open
  aborts were nearly absent.
- Audited asymmetric arenas and established that raw trajectory \(y\), stored
  map rows, and paper partition rows all use image-style north-to-south order.
  Corrected raw seam frames, introduced-boundary normals, accessible-support
  indexing, documentation, and regression tests. Stored seam-strip bins and
  results were numerically unchanged because the previous two inversions had
  canceled.
- Re-ran the exact oriented-seam screen. Shared-wall cell-edit correlations
  exceeded wall/open correlations in all seven mice and in horizontal and
  vertical subsets.
- Independently stress-tested the result. It survives cell-label permutation,
  raw target rates without square subtraction, equal-bin and occupancy
  weighting, session-global-rate removal, and a same-signed-normal
  different-seam control in 6/7 mice (all 6/6 coverage-eligible). It is
  positive within 10 cm and
  approximately zero 17.5–22.5 cm from the wall.
- Implemented target-rate-held-out local prediction: learn an exact-location
  wall-conditioned profile from every other geometry in exposure \(k\),
  withhold the target neural rates, and compare it with the target local cell
  vector in exposure \(k+1\) using an independent outer square baseline. The
  target wall label, occupancy, and registration support are test-aware.
  Exact-wall beats exact-location open in all seven mice;
  the global-rate-demeaned animal mean advantage is 0.214.
- Reconstructed the within-cycle result from contiguous raw-event half-session
  maps. All 14/14 coverage-eligible exposure blocks and all 5/5 eligible mice
  are positive; a stricter dwell threshold retains 10/10 blocks in 4/4 mice.
  Recombination reproduces released maps to machine precision with zero
  finite-mask mismatches after blocked-tile rate masking.
- Estimated raw-event spatial rates at a common familiar-square-calibrated
  reference for speed, allocentric movement direction, and linear session
  time. The target-rate-held-out local advantage remains positive in 7/7 mice
  (animal mean 0.209); omitting session time gives 7/7 positive and mean 0.210,
  with no sign flips. The adjustment is additive and does not include local
  interaction matching.
- Reweighted the target-rate-held-out records equally by exposure pair, target
  environment, and exact oriented seam. Raw and global-rate-demeaned
  advantages remain positive in 7/7 mice under every scheme; the latter
  cohort means range from 0.209 to 0.221.
- Ran the symmetric reversal for target seams known to be open. After
  global-rate demeaning, open profiles beat wall profiles in 7/7 mice (animal
  mean 0.049); raw-rate reversal is positive in 6/7 (mean 0.051). The
  target-aware descriptive control has 815 eligible directed local queries,
  with sparse support in Q74 and Q75.
- Replaced the liberal within-record identity shuffle with 999 coherent
  animal-level global-ID mappings shared across all dependent records and
  target-side sessions while preserving each common-cell set. Global-rate-
  demeaned one-sided diagnostic tails are at most 0.005 in every mouse; raw
  Q51 does not pass (0.167). Exchangeability is constrained, so these remain
  identity diagnostics rather than population inference.
- Tested the randomized first-cycle shape order for accelerated structural
  transfer. Prior matching-seam coverage did not predict 5- or 10-minute map
  maturity after geometry, serial-position, whole-shape, behavior, and
  recording controls. Recurring exact-location profiles therefore do not imply
  zero-shot or faster learning.
- Tested whether two exact-wall profiles add linearly at a corner. The design
  contains only two eligible orthogonal co-wall configurations, both in `bit
  donut`; the additive predictor did not beat the best individual-wall
  template and the branch was stopped as underidentified.
- Added a focused discovery report, tracked cross-exposure and raw-event
  source-data artifacts, deterministic stored-map controls, a compact
  evidence-chain figure/source table, and new unit tests.
- Installed the locked dependencies and project into a new virtual
  environment; all 86 tests passed there and in the working environment, with
  no broken requirements. The temporary environment was removed after the
  check.
- Re-ran target-rate-held-out prediction after exactly matching every
  non-focal neighbor state of the accessible target partition. The
  global-rate-demeaned wall/open advantage remained positive in 7/7 mice
  (mean 0.184; 252 predictions), independently for horizontal and vertical
  focal seams.
- Isolated the `u` versus `o` counterfactual, in which the complete blocked-tile
  vectors differ only at partition 5, and transferred its edit to held-out
  `i`, `l`, and `t` sessions. The same-location advantage was 0.225 and
  positive in 7/7 mice; raw-event adjustment for speed, allocentric movement
  direction, and session time preserved the result in 7/7.
- Audited the perfect `u`-after-`o` order. A generic same-seam
  later-minus-earlier vector was negative on average and positive in 0/7,
  while the drift-adjusted `u-o` vector remained predictive in 7/7. This
  contradicts simple generic sequence drift but cannot remove an
  environment-specific order interaction.
- In a separate estimator from the `u-o` endpoint test, reconstructed a
  multi-geometry wall-minus-open vector at a different, non-overlapping seam
  exactly one 25 cm grid step away with the same signed normal. The translated
  effect predicted the held-out target wall residual in 7/7 mice (mean
  \(r=0.148\)), was specific relative to a training-derived target-open
  profile, and was weaker than the query-matched exact-location benchmark
  (\(r=0.230\)) by 0.082.
- Audited the transfer spatially. Source and target strips share no 5 cm bins,
  but layout-matched controls do not support a general direction-selectivity
  claim. Coherent cell-identity permutations support the effect strongly in
  five animals but not uniformly in Q51 and Q75.
- Reconstructed the cross-location test from raw events after additive
  adjustment for speed, allocentric movement direction, and session time.
  The translated effect remains positive in 7/7 mice (mean \(r=0.119\));
  specificity relative to the training-derived target-open profile is 0.172,
  also positive in 7/7. The audit matches stored-map query/session/seam
  identities and common-cell/bin counts across 405 target queries and 658
  source pairs.
- Froze a same-session mirror-open spatial-lag control before neural scoring.
  It matches source distance, translation axis, strip gap, relative bins,
  cells, session, and square baseline. For 72 supported tangential triplets in
  five animals, the demeaned wall-minus-mirror advantage is 0.114 and positive
  in 5/5. Absolute and raw-rate signs are not uniform, Q74/Q75 lack support,
  and normal-axis translations are structurally untestable, so the result only
  narrows generic symmetric nearby continuity.
- Tested additive composition. Its primary demeaned correlation was below the
  best single-wall predictor in 4/5 eligible animals (mean difference -0.024),
  and its normalized error was worse in 5/5. The full geometry design is
  rank-deficient for an oriented-wall additive model.
- Tested experience against same-support open/open reliability and
  non-overlapping raw halves. Strict raw coverage collapsed to two animals,
  and the stricter sensitivity was discordant; no learning claim was
  promoted.
- Locked before execution and ran a former-barrier scar prediction in the
  independent Blair et al. neutral-barrier dataset. The effect was negative
  on average and positive in only 1/6 rats on the locked primary
  Pearson-active metric; the Spearman-active sensitivity was approximately
  zero and positive in 3/6. Persistent barrier memory was rejected without
  retuning.
- Consolidated the surviving and failed predictions into
  `reports/local_wall_update_transfer.md`, a deterministic evidence figure,
  source tables, dedicated audit reports, and regression tests. The lead is
  limited reuse and partial transfer of a local CA1 wall-related update—not a
  generic mental map, portable wall symbol, additive grammar, learning, or
  innateness.
- Regenerated the four-panel stronger-claim figure, validated all 32 tracked
  source JSON artifacts, compiled every Python source file, and passed the
  complete 147-test working-tree suite.
- Replaced the earlier boundary-normal LaTeX draft with an 11-page active
  manuscript on limited spatial transfer of a cell-resolved CA1 wall-related
  update. Added a significance statement, evidence ladder, formal methods,
  complete limitation ledger, prospective experiment, and updated primary
  literature. The final PDF compiled without document warnings and every page
  was rendered and visually inspected.
- Added Javier Emilio Bazan Sanchez as author, with Facultad de Ciencias,
  UNAM affiliation and institutional email. Recompiled and visually audited
  the resulting 12-page PDF. Published a curated public scientific repository
  at `https://github.com/asim-v/ca1-wall-update-transfer` with derived source
  data, manuscript source and PDF, analysis scripts, package code, tests,
  CITATION metadata, and explicit raw-data exclusions. The public subset
  passes 146 tests with one raw-data-dependent skip.
- Prepared a minimal arXiv source package with `main.tex`, `main.bbl`,
  `references.bib`, and the required PNG figure. The exact extracted ZIP
  recompiles to an 11-page PDF without document warnings and passed a complete
  visual audit. Added ready-to-paste arXiv metadata for q-bio.NC. Account
  registration awaits the author's password, CAPTCHA authorization, and
  confirmation of personal and legal declarations.
- Added an exact animal-level spatial-null calibration over all sign
  assignments. The pooled equal-midpoint wrong-orientation advantage is 0.101
  and positive in 7/7 animals, whereas the fully strip-matched tangential
  sensitivity is positive in only 3/6. This sharpens the supported finding to
  partial, spatially dependent generalization without promoting a general
  orientation-specific mechanism.
- Reframed the manuscript around the positive empirical result. The new title,
  abstract, significance statement, evidence table, and discussion lead with
  cross-location generalization and its attenuation relative to exact reuse;
  secondary negative branches now appear in a supplementary exploratory
  section.
- Recompiled and visually inspected all 11 pages of both the working PDF and
  arXiv-targeted PDF. Rebuilt the four-entry arXiv ZIP, compiled its exact
  extracted contents, and verified pixel-identical rendered pages. The package
  SHA-256 is
  `B18FCD73E39614EA32564799C73B69CF70A8165F1E4A443517DE3C041502645A`.
- Completed the arXiv `neuroemi` account registration and institutional-email
  verification. Submission still requires password recovery or reset, login,
  personal declarations, a license choice, and review of the server-generated
  PDF before the irreversible final submission action.
- Published the revised manuscript, spatial-null report and JSON artifact,
  deterministic summarizer, figure, and regression test to the public
  reproducibility repository in commit `04f1fc1`.
- Converted the manuscript from a single-column 11-page draft to an 8-page
  two-column layout. Both figures and the evidence table span the full page
  width, and the bibliography is balanced across its final page.
- Removed rhetorical stage directions such as "we asked the stronger
  question," "transfer was incomplete," and "the decisive next experiment is
  compact." Replaced them with direct descriptions of source-to-target
  prediction, the exact-location benchmark, controls, and prospective design.
- Rebuilt the four-entry arXiv ZIP from the revised source and compiled its
  exact extracted contents. The new package SHA-256 is
  `9F48D7C11FDB59FBE055B7928E6D031F9AB0A1D8585FC2C73310A5FA4C2AC96D`.
- Published the two-column manuscript, updated figure labels, and simplified
  prose to the public reproducibility repository in commit `c878474`.
- Added a four-panel explanatory figure that shows the 25 cm source-to-target
  test, the paired exact-versus-cross-location gradient, the pooled and strict
  orientation controls, and the same-session mirror-open control. Every plotted
  value is generated from tracked JSON artifacts and checked by regression
  tests. The working and exact extracted arXiv PDFs remain 8 pages and passed a
  complete visual audit. The five-entry arXiv ZIP is 657,056 bytes with SHA-256
  `5259FB5B82582535168A0B61D73077060224103CB3778DB48B6481188B2C8FCF`.
- Published the explanatory figure, its source-data CSV, deterministic
  generator, regression checks, and revised manuscript PDF to the public
  reproducibility repository in commit `9fb3f0d`.
- Reworked Methods Section 4.3 so the familiar-square subtraction and the
  registered-cell response vector appear as a single aligned equation. Replaced
  the visually opaque square-session operator with the explicit definition
  \(b(e)\), simplified the surrounding prose, retained the 8-page layout, and
  visually audited both the working PDF and the exact extracted arXiv package.
  The revised ZIP is 657,119 bytes with SHA-256
  `A5DFB4076D53B67C38AE4ACB045DEC8E3F910EA6D75A8EB600F9E65D1D6006E1`.
- Published the clarified notation and rebuilt manuscript PDF to the public
  reproducibility repository in commit `11b7b1d`.
- Replaced the brief cohort paragraph with a formal Dataset description. It now
  states the seven-animal, 5,413-cell, 207-session scope; the familiar square
  and nine internal-wall layouts; the per-animal exposure-cycle counts; the
  released data modalities; and the fixed-order confound. Compact back-matter
  headings and a smaller bibliography preserve the 8-page two-column layout.
  The exact extracted arXiv package was visually audited on all pages. Its ZIP
  is 657,265 bytes with SHA-256
  `254906876768A8D2F243D9DE99A0CFA892CE63D8F8D900C404EB01958E54618F`.
- Published the formal dataset description and rebuilt manuscript PDF to the
  public reproducibility repository in commit `b759cc3`.
- Fused the wall-generalization manuscript with the independently developed
  reward-relocation analysis. The new paper leads with reuse of a registered-
  cell population-change direction across absolute locations, while keeping
  the two estimators and inferential statuses separate.
- Added the eight-mouse reward specificity result, deconvolved-event
  replication, registered pre-switch and fixed-session drift diagnostics,
  trial and transition-label nulls, preprocessing and registration
  sensitivities, corrected speed/lick geometry, and the frozen nonlinear
  out-of-fold trial model.
- Integrated two reward figures and expanded the evidence table, Dataset
  description, Methods, Discussion, and prospective experiment. The final
  two-column PDF is 13 pages with five figures and one table; every page was
  rendered and visually inspected. The boundary and reward repositories pass
  151 and 47 tests, respectively.
- Built `submission/ca1_reusable_population_updates_arxiv.zip` with seven
  source files and retained the wall-only ZIP as a backup. The fused ZIP is
  1,142,647 bytes with SHA-256
  `EC2AB1AA638BE6F91944DA4992A2A44D4542751B5F1FAA178005ED5C631C2461`.
- Published the reward-analysis companion at
  `https://github.com/asim-v/ca1-goal-update-transfer`, containing frozen
  controls, compact derived results, registration audits, scripts, and 47
  passing regression tests without source recordings.
- Replaced provisional code-availability language with permanent URLs for both
  public repositories. Recompiled the 13-page PDF and visually verified the
  affected data-availability page.
- Rebuilt the arXiv source archive with the required `figures/` directory and
  compiled its exact extracted contents. The corrected ZIP is 1,141,988 bytes
  with SHA-256
  `BA49D3B2771F744F8B8C492D57276E5D616F8FECAF6BCE9417B2E2432F6FC59E`.
- Pushed the fused boundary-repository release as commit `7d122e1` on branch
  `codex/fused-manuscript-release` and opened draft PR
  `https://github.com/asim-v/ca1-wall-update-transfer/pull/1` for review before
  changing the public default branch.
- Marked PR #1 ready and squash-merged it into public `main` as
  `c9a5d794293acfdd76c99d704ee98265c99d9cb4`. GitHub now serves the fused
  README, 1,270,805-byte manuscript PDF, and 1,141,988-byte arXiv source ZIP
  from the default branch.
- Removed the standalone Significance statement from both active LaTeX sources.
  The abstract and keywords now lead directly into the Introduction.
- Recompiled and visually inspected all 12 pages. The new PDF SHA-256 is
  `7152BB184859F782B829F87C917D700B5B1C4FE8F220F1B5372A62BD18FFB7B4`.
- Rebuilt the exact seven-file arXiv archive and compiled it after clean
  extraction. The 1,141,803-byte ZIP has SHA-256
  `12260E50DB54CF36C3AAD054681EF7CCA6053AD4A7134A33EA55494FF2F649A1`.
- Pushed commit `4606501` on `codex/remove-significance-statement` and opened
  draft PR `https://github.com/asim-v/ca1-wall-update-transfer/pull/2`.
- At the author's request, pushed `4606501` directly to public `main`. GitHub
  marked PR #2 merged because the base branch now contained its exact head.
  Future authorized public updates will use direct `main` pushes unless the
  author explicitly asks for a PR.
## 2026-08-02 - Specificity revision and empirical positive baseline

- Audited repository provenance, quantitative claims, current primary
  literature, and novelty positioning before running a new endpoint.
- Froze and committed the positive-model adjudication protocol, then
  implemented the exhaustive exact-distance source rank.
- Reconstructed 415 enumerated queries, 405 inherited primary queries, and 658
  correct-relation source pairs. The primary percentile rank was 0.612 and
  above chance in 7/7 animals; correct-minus-mean-alternative was positive in
  7/7, while correct-minus-best was positive in 5/7.
- Applied the protocol stop to the fitted M0--M3 arm because the empirical
  weight could not be calibrated on neural sessions disjoint from every outer
  test pair. No partial fitted target endpoint was inspected.
- Reframed the manuscript around partial registered-cell transfer, the 0.082
  shifted-versus-exact decrement, and behavior-shared reward geometry.
- Added a full-enumeration figure, sentence-level quantitative abstract
  traceability, current primary references, and an explicit fitted-model
  identifiability statement.
- Built a 14-page two-column PDF, rendered and inspected all pages, rebuilt an
  eight-entry arXiv ZIP, and compiled the exact extracted archive.
- Renamed the reward robustness plot to transition-class language and pushed
  companion commit `61d76a5` directly to public `main`; numerical results were
  unchanged.
