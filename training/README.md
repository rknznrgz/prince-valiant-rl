# Trained models and fitting logs

Selected experiment artifacts, distinct from reusable code in [src/prince_valiant_rl/training/](../src/prince_valiant_rl/training/).

| Directory | Contents |
| --- | --- |
| `level1_dqfd_demo_continuation_20261025` | Selected update-3,000 DQfD checkpoint and committed receipt |
| `level1_condition_B_score_quarantine_v1_20261024` | Preserved configuration for the selected Level 1 lineage |
| `level2_title_bc_v57_20261026` | Selected Level 2 imitation actor and fitting records |
| `level3_imitation_v1_20261026` | Initial Level 3 fitting records |
| `level3_imitation_refine_v2_20261026` | Selected refined Level 3 actor and fitting records |

Run `python run.py inspect_models` to validate included model hashes and state shapes on CPU. `fit.jsonl` records supervised fitting diagnostics, not episode returns. See [results](../docs/RESULTS.md) and [required assets](../manifests/artifacts.json).
