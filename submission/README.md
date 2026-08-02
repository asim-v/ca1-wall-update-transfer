# arXiv submission package

The upload-ready fused source package is generated from `submission/arxiv/`
and contains:

- `main.tex` - top-level two-column LaTeX source;
- `main.bbl` and `references.bib` - bibliography artifacts;
- `figures/local_wall_update_transfer.png` - boundary reuse results;
- `figures/partial_generalization_explainer.png` - boundary transfer logic and
  spatial controls;
- `figures/adversarial_controls_v1.png` - reward drift, null, and parameter
  controls;
- `figures/trial_behavior_model_v1.png` - nonlinear trial-level speed and
  licking analysis.

The source package intentionally excludes the generated PDF, logs, auxiliary
files, raw data, and unrelated repository content. The previous wall-only ZIP
is retained as a backup.

Current fused package:

```text
submission/ca1_reusable_population_updates_arxiv.zip
```

- Entries: 7 files plus the `figures/` directory.
- Size: 1,141,803 bytes.
- SHA-256: `12260E50DB54CF36C3AAD054681EF7CCA6053AD4A7134A33EA55494FF2F649A1`.
- Validation build: 12 letter-size pages, 5 figures, and 1 evidence table.

The `neuroemi` arXiv account and institutional-email verification were
previously completed. Login, author declarations, category endorsement,
server-side compilation, metadata preview, license selection, and the final
Submit Article action remain external submission gates.
