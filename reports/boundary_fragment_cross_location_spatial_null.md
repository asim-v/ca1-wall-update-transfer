# Cross-location spatial-null calibration

## Question

Does the observed cross-location prediction exceed empirical controls that
retain the recorded CA1 maps, place-field smoothness, registered-cell
identities, and source-target lag while breaking the correct wall relation?

## Nulls

The calibration uses animals as the independent units and enumerates every
possible sign assignment for each paired animal-level contrast. Repeated
queries, seams, and cells are not treated as independent replicates.

1. **Equal-midpoint wrong orientation.** Correctly oriented and incorrectly
   oriented source seams are both 25 cm from the same held-out wall target.
   This retains the observed spatial maps and target response while breaking
   the correct signed wall relation.
2. **Tangential 5 cm strip lag.** The comparison is restricted to tangential
   translations with the same 25 cm midpoint distance and 5 cm nearest-strip
   separation.
3. **Same-session reflected open.** The same source predictor is evaluated
   against a wall and a reflected open seam in the same target session, with
   translation axis, source distance, relative bins, occupancy support, and
   registered cells matched.

## Results

For globally demeaned rates:

| Control contrast | Mean | Positive animals | Exact one-sided sign fraction |
| --- | ---: | ---: | ---: |
| Correct minus wrong orientation, 25 cm midpoint | 0.101 | 7/7 | 1/128 |
| Correct minus wrong orientation, tangential 5 cm strip lag | 0.040 | 3/6 | 0.25 |
| Correct minus wrong orientation, normal axis | 0.138 | 7/7 | 1/128 |
| Wall minus reflected open, same session | 0.114 | 5/5 | 1/32 |

The pooled equal-midpoint and reflected-open advantages remained positive in
every leave-one-animal-out summary. Raw rates reproduced the pooled
equal-midpoint advantage (0.106, 7/7), but the reflected-open advantage was
less consistent (0.079, 3/5).

## Interpretation

The cross-location result is not an arbitrary correlation between any two
seams with the same midpoint separation: the correct wall relation wins in the
pooled comparison, and the transferred contrast distinguishes a wall from an
equally displaced open location on the strict supported subset.

The null does not completely eliminate smooth spatial continuity. The most
tightly distance-matched tangential orientation contrast is heterogeneous, and
the reflected-open result covers only five animals and demeaned tangential
translations. The strongest justified conclusion is therefore **partial
cross-location generalization with residual spatial ambiguity**, not a freely
portable or direction-invariant wall mechanism.

Machine-readable source:

- `results/source_data/boundary_fragment_cross_location_spatial_null.json`
