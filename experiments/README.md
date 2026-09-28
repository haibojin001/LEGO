# Experiments

Each directory holds `experiment.yaml` (arms, sweep, tables) and a README with the exact commands. Identical arm configurations are shared across experiments (records are keyed by arm id), so an arm listed in several experiments runs once.

| experiment | RQ | what | arms x sweep |
|---|---|---|---|
| [`rq1_scratch`](rq1_scratch/) | RQ1 | Construction from scratch across 13 backbones | 1 x 13 |
| [`rq1_single_pass`](rq1_single_pass/) | RQ1 | Single pass vs. execution feedback | 2 x 1 |
| [`rq1_external_agents`](rq1_external_agents/) | RQ1 | Independently developed repository agents | 2 x 1 |
| [`rq2_library`](rq2_library/) | RQ2 | CodeFace reach and the runtime adaptation funnel | 1 x 1 |
| [`rq2_library_mix`](rq2_library_mix/) | RQ2 | Library source, size and retrieval depth | 12 x 1 |
| [`rq3_paradigms`](rq3_paradigms/) | RQ3 | Construction paradigms | 9 x 1 |
| [`rq3_ablation`](rq3_ablation/) | RQ3 | Construction-stage attribution | 5 x 1 |
| [`rq3_consumption`](rq3_consumption/) | RQ3 | How reused components are consumed | 5 x 1 |
| [`rq3_fields`](rq3_fields/) | RQ3 | Primitive-field decomposition | 6 x 1 |
| [`rq3_provenance`](rq3_provenance/) | RQ3 | Donor provenance and contamination control | 6 x 1 |
| [`rq3_components`](rq3_components/) | RQ3 | Component removals | 6 x 1 |
| [`rq4_backbones`](rq4_backbones/) | RQ4 | Backbone substitution | 2 x 13 |
| [`rq4_resident_swap`](rq4_resident_swap/) | RQ4 | Adaptation-model substitution | 1 x 9 |
| [`rq4_diagnosis_swap`](rq4_diagnosis_swap/) | RQ4 | Diagnosis-model substitution | 1 x 9 |
| [`rq5_allocation`](rq5_allocation/) | RQ5 | Quality-cost trade-off under model allocation | 11 x 1 |
| [`rq5_budget`](rq5_budget/) | RQ5 | Attempt-budget ladder | 3 x 1 |
| [`rq6_transfer`](rq6_transfer/) | RQ6 | Transfer to external benchmarks | 2 x 1 |
| [`app_replicates`](app_replicates/) | Appendix | Replicate runs | 6 x 3 |
| [`app_strata`](app_strata/) | Appendix | Contamination strata (star count) | 2 x 1 |
| [`app_band_sensitivity`](app_band_sensitivity/) | Appendix | Difficulty-band sensitivity and scoring checks | 3 x 1 |
| [`app_special_casing`](app_special_casing/) | Appendix | Test-targeted special-casing audit | 2 x 1 |
| [`codeface_harvest_seed`](codeface_harvest_seed/) | Artifact | Separate construction pass supplying harvest candidates | 1 x 1 |
