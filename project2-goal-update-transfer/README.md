# Pinned reward-result mirrors

This directory contains the two machine-readable reward-arm result files used
by the fused manuscript abstract. They are byte-for-byte mirrors of files from
[`asim-v/ca1-goal-update-transfer`](https://github.com/asim-v/ca1-goal-update-transfer)
at source commit `2d97ee7262d540595019217d98c26ae8701a64e2`.

They are included so that
`scripts/validate_abstract_claim_traceability.py` works in a standalone clone
of this repository. The authoritative protocols, implementation, full results,
and history remain in the companion repository.

Pinned SHA-256 values:

- `results/confirmatory_specificity_v1.json`:
  `4C9A85B130B22E6E0428CD9F0F9D3586F62A89ED4A799ECB36D70A1AF1E1C47E`
- `results/trial_behavior_model_v1.json`:
  `36C2C813467F9158C47993D33F9640351DC97BC6FFE456247E68A67743855B12`
