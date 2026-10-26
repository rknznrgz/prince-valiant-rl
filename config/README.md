# Configuration

Versioned RAM maps, action/model contracts, reward selection, and experiment settings used by the retained workflows.

`backward_level1_v1.json` is the base PPO example. The selected Level 1 DQfD lineage uses its preserved [experiment configuration](../training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json) and checkpoint metadata.

Keep configuration versions paired with their model, schema, and dataset identities. Changing reward weights requires consistent cached-target relabeling; an evaluation-only change is not a new training result.
