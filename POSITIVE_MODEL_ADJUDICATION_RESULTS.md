# Positive-model adjudication results

Result date: 2026-08-02

Protocol commit: `dac65b4ca608839ffbb12315546806ddea4ba512`

Implementation commits: `bf7d438a9b18ee7775e483828bf07b78e1908fda`,
`d992fe6623612197c50e5feeedbdb1e78f2b9b69`, and
`ef1208a483c1eb1c57a2f05e1736dc365adc0ea1`

Machine-readable result:
`results/positive_model_spatial_rank_v1.json`

Classification: **frozen-local empirical positive baseline**. The wall data
were explored before this protocol; this is not a prospective confirmation of
the broader wall hypothesis.

## Verdict

When every geometry- and support-admissible source seam exactly 25 cm from a
target was enumerated, the correct same-signed-normal relation ranked above
chance in all seven animals. It also outperformed the mean alternative in all
seven. The result was weaker against the best alternative and in the strict
tangential subset. Thus the result survives a full empirical spatial baseline,
but does not establish that the empirical update contains information beyond a
fitted boundary-vector/place-field model.

## Ledger reconstruction and coverage

The runner verified the frozen source ledger hash and exactly reconstructed:

- 415 enumerated target-query records;
- 405 primary one-grid-step same-signed-normal queries;
- 658 primary correct-relation source pairs.

The Tier-1 rank additionally requires at least one exact-distance alternative,
so its scored-query count is smaller than the inherited primary count. Counts
below are descriptive support, not biological replicates.

| Animal | Enumerated queries | Primary queries | Correct source pairs | Tier-1 scored queries (pairs) | Tangential scored queries (pairs) |
|---|---:|---:|---:|---:|---:|
| QLAK-CA1-08 | 81 | 81 | 136 | 79 (2) | 45 (2) |
| QLAK-CA1-30 | 82 | 82 | 138 | 80 (2) | 46 (2) |
| QLAK-CA1-50 | 82 | 82 | 138 | 80 (2) | 46 (2) |
| QLAK-CA1-51 | 41 | 41 | 69 | 40 (1) | 23 (1) |
| QLAK-CA1-56 | 82 | 82 | 138 | 80 (2) | 46 (2) |
| QLAK-CA1-74 | 13 | 10 | 10 | 6 (2) | not estimable |
| QLAK-CA1-75 | 34 | 27 | 29 | 22 (2) | 3 (1) |

All target, source, cell, bin, occupancy, and neural-session gates were those
frozen in the protocol. No source was selected by neural similarity. Queries
were averaged within exposure pair and exposure pairs within animal before
cohort summaries and exact sign flips.

## Tier 1: all admissible exact-25-cm sources

The primary endpoint is the percentile rank of the equal-weight correct-
relation score among every admissible source score, centered at chance (0.5).
Therefore `+0.1118` corresponds to a mean animal-level percentile rank of
`0.6118`, not to a correlation coefficient.

### Frozen primary rate mode: global-rate-demeaned

| Endpoint | Animal mean | Positive animals | Exact one-sided P | Exact two-sided P | Interpretation |
|---|---:|---:|---:|---:|---|
| Centered percentile rank | 0.111782 | 7/7 | 0.0078125 | 0.015625 | primary endpoint passes |
| Correct minus mean alternative correlation | 0.101934 | 7/7 | 0.0078125 | 0.015625 | correct relation exceeds the full alternative mean |
| Correct minus best alternative correlation | 0.059184 | 5/7 | 0.078125 | 0.15625 | heterogeneous; does not pass |
| Correct-win probability minus 0.5 | 0.144826 | 5/7 | 0.0625 | 0.125 | suggestive, not decisive |

Primary centered-rank animal values:

| Animal | Centered rank | Percentile rank |
|---|---:|---:|
| QLAK-CA1-08 | 0.019282 | 0.519282 |
| QLAK-CA1-30 | 0.107083 | 0.607083 |
| QLAK-CA1-50 | 0.259792 | 0.759792 |
| QLAK-CA1-51 | 0.083125 | 0.583125 |
| QLAK-CA1-56 | 0.062500 | 0.562500 |
| QLAK-CA1-74 | 0.166667 | 0.666667 |
| QLAK-CA1-75 | 0.084028 | 0.584028 |

### Raw-rate sensitivity

| Endpoint | Animal mean | Positive animals | Exact one-sided P | Exact two-sided P |
|---|---:|---:|---:|---:|
| Centered percentile rank | 0.104922 | 7/7 | 0.0078125 | 0.015625 |
| Correct minus mean alternative correlation | 0.106329 | 7/7 | 0.0078125 | 0.015625 |
| Correct minus best alternative correlation | 0.060129 | 4/7 | 0.148438 | 0.296875 |
| Correct-win probability minus 0.5 | 0.127610 | 5/7 | 0.054688 | 0.109375 |

The raw sensitivity reproduces the primary rank and mean-alternative pattern,
while preserving the failure against the best alternative.

## Tier 2: strict tangential translation

This tier moves along the wall axis and therefore better retains local
distance-to-wall geometry, but only six animals are estimable and one of those
has three scored queries. It was frozen as support-limited.

### Global-rate-demeaned

| Endpoint | Animal mean | Positive animals | Exact one-sided P | Exact two-sided P |
|---|---:|---:|---:|---:|
| Centered percentile rank | 0.071421 | 4/6 | 0.078125 | 0.15625 |
| Correct minus mean alternative correlation | 0.040431 | 3/6 | 0.25 | 0.50 |
| Correct minus best alternative correlation | 0.015375 | 3/6 | 0.390625 | 0.78125 |
| Correct-win probability minus 0.5 | 0.094971 | 4/6 | 0.1875 | 0.375 |

### Raw-rate sensitivity

| Endpoint | Animal mean | Positive animals | Exact one-sided P | Exact two-sided P |
|---|---:|---:|---:|---:|
| Centered percentile rank | 0.079916 | 5/6 | 0.046875 | 0.09375 |
| Correct minus mean alternative correlation | 0.024903 | 3/6 | 0.34375 | 0.6875 |
| Correct minus best alternative correlation | -0.000029 | 3/6 | 0.515625 | 1.0 |
| Correct-win probability minus 0.5 | 0.116875 | 4/6 | 0.140625 | 0.28125 |

The isolated raw centered-rank one-sided tail does not override the frozen
demeaned primary rate mode, the 4/6 sign pattern, or the negative/heterogeneous
companion endpoints. The correct conclusion remains that the strict tangential
test is unresolved.

## Fitted M0–M3 boundary model: protocol stop

The raw wall files contain the maps, occupancy, geometry, and registration
needed to fit M0 and M1. The primary M1-to-M3 adjudication nevertheless fails a
predeclared design condition: M3's stack coefficient must be identified from a
training-only pseudoquery that reuses neither neural session in the outer test
pair.

Six animals have three exposure cycles, yielding only adjacent pairs 1→2 and
2→3. For the first outer pair, the only candidate pseudo-pair reuses cycle 2,
which contains outer target neural sessions. QLAK-CA1-51 has only two exposure
cycles and therefore no independent pseudo-pair at all. Consequently:

- an M3 coefficient cannot be trained leakage-free for every outer pair;
- QLAK-CA1-51 cannot receive any M3 prediction;
- the protocol requires all seven animal summaries;
- fitting M0/M1 and inspecting partial M1–M3 targets would violate the stopping
  rule rather than provide a valid reduced analysis.

No M0, M1, M2, or M3 target endpoint was calculated. The required design is at
least one additional independently usable exposure cycle per animal, or a
separate development cohort in which M2/M3 calibration is fixed before these
seven targets are scored. An alternative future protocol could freeze a
calibration-free residual statistic, but it cannot be substituted post hoc in
version 1.0.

## Reward R0–R4 status

The reward positive-model family was frozen as a secondary extension but was
not executed in this result stage. The already tracked nonlinear behavior model
is not relabeled R3: it conditions neural trials on speed and licking but does
not include the protocol's joint track-relative and reward-relative positive
model. No new reward endpoint was inspected.

The reward arm therefore retains its existing conclusion: substantial
behavior-shared transition geometry and a smaller residual registered-cell
relation. Whether a joint position/reward/behavior model exhausts that residual
remains unresolved.

## Interpretation

### What became stronger

The correct wall relation was not compared with one favorable control. It
ranked above the fully enumerated set of exact-distance admissible sources in
all seven animals, under both demeaned and raw rates. This directly weakens the
claim that the original result depended on choosing a convenient incorrect
orientation.

### What did not become stronger

The correct relation did not reliably beat the best available alternative, and
the strict tangential tier remained heterogeneous. The fitted boundary-positive
model could not be evaluated under its frozen leakage barrier. Therefore these
results support descriptive relation-specific reuse, but not model-level reuse
beyond boundary-relative or smooth-place tuning.

Permitted wording:

> Across all exact-distance admissible sources, the correct wall relation had
> an above-chance prediction rank in every animal, although it did not
> consistently outperform the best alternative and the strict tangential
> subset remained heterogeneous.

Forbidden wording:

> The positive-model analysis proves a portable wall-update operator or rules
> out boundary-vector and spatial-continuity explanations.
