# Fused boundary-and-reward manuscript build

Date: 2026-08-02

## Scientific organization

The active manuscript combines two independent longitudinal CA1 reanalyses
under one question: does the direction of a registered-cell population change
recur when a spatial relation is repeated at a different absolute location?

The numerical endpoints are not pooled.

- Boundary dataset: wall-minus-open contrasts predict the same seam and a
  non-overlapping seam 25 cm away. Cross-location prediction is positive in
  7/7 mice and lower than the matched exact-location benchmark.
- Reward dataset: two same-signed 120 cm relocations have more aligned update
  fields than an opposite-signed 240 cm return. The frozen dF/F endpoint is
  positive in 8/8 confirmatory mice. A nonlinear out-of-fold speed and lick
  model leaves residual dF/F specificity in 7/8 while reducing its cohort mean
  to 49.5% of the original.

The supported synthesis is structured reuse with spatial and behavioral
dependence. The manuscript does not claim a coordinate-free operator,
behavior independence, symbolic computation, portable amplitude, composition,
innateness, or a universal hippocampal primitive.

## Deliverables

- LaTeX source: `manuscript/main.tex`
- Bibliography: `manuscript/references.bib`
- Audited PDF: `output/pdf/ca1_reusable_population_updates_manuscript.pdf`
- arXiv metadata: `submission/ARXIV_METADATA.md`
- arXiv source: `submission/arxiv/`
- arXiv ZIP: `submission/ca1_reusable_population_updates_arxiv.zip`

## Build verification

- Compiler: Tectonic 0.16.9.
- Layout: letter paper, two columns, 12 pages.
- Front matter: abstract and keywords proceed directly to the Introduction;
  the separate Significance statement has been removed.
- Contents: 5 figures and 1 full-width evidence table.
- Bibliography: 12 entries; all citations and cross-references resolved.
- Visual audit: all 13 rendered pages inspected; no clipped figures, tables,
  equations, URLs, headings, or references.
- PDF forms, encryption, and JavaScript: none.
- Regression suites: 150 boundary-repository tests passed, one was skipped by
  design, and all 47 reward-repository tests passed.
- PDF SHA-256:
  `7152BB184859F782B829F87C917D700B5B1C4FE8F220F1B5372A62BD18FFB7B4`.

Tectonic reports narrow-column underfull-box diagnostics and a 1.25 pt final
balanced-column vbox diagnostic. The rendered pages show no overlap or clipping.

## arXiv package verification

The ZIP contains only:

1. `main.tex`
2. `main.bbl`
3. `references.bib`
4. `figures/local_wall_update_transfer.png`
5. `figures/partial_generalization_explainer.png`
6. `figures/adversarial_controls_v1.png`
7. `figures/trial_behavior_model_v1.png`

Package size: 1,141,803 bytes.

Package SHA-256:
`12260E50DB54CF36C3AAD054681EF7CCA6053AD4A7134A33EA55494FF2F649A1`.

The package compiled locally to the same 12-page PDF. Generated PDF, AUX, and
OUT files are excluded from the ZIP. The earlier wall-only ZIP remains in
`submission/` as a recoverable backup.

## Public release status

- Boundary repository: `https://github.com/asim-v/ca1-wall-update-transfer`
- Reward repository: `https://github.com/asim-v/ca1-goal-update-transfer`
- The manuscript data-and-code statement contains both permanent URLs.

## Remaining submission gates

1. Confirm funding disclosures, CRediT wording, arXiv license, and author-only
   declarations.
2. Upload the source ZIP, inspect the server-generated PDF, and submit only
   after the metadata and rendered document match the local audit.
