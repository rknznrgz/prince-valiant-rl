# Reproducibility and verification

## Included artifacts

The repository contains environment, training, evaluation and preparation code; three selected checkpoints; authentic selected metrics; a 10-page draft report; and four videos.

| Model | Checkpoint |
| --- | --- |
| Level 1 DQfD, update 3,000 | `training/level1_dqfd_demo_continuation_20261025/checkpoint_3000.pt` |
| Level 2 title-inclusive imitation | `training/level2_title_bc_v57_20261026/actor.pt` |
| Level 3 refined imitation | `training/level3_imitation_refine_v2_20261026/best.pt` |

[Artifact records](../manifests/artifacts.json) contain SHA-256 hashes, byte sizes, access conditions, and destinations. The Level 1 committed receipt preserves model/optimizer/replay/RNG identities. Levels 2 and 3 include normalization buffers and action vocabularies in their checkpoints.

Recorded seeds are 20261024 for selected DQfD training, 20261128 for Level 2 fitting, and 20261130 for Level 3 fitting. The base PPO config uses 20261015; the selected Level 1 lineage config uses 20261022. These apply to their respective experiments, not a single shared run.

## Verification performed

[verification.json](../manifests/verification.json) records:

- Compilation checks and help checks for all 16 advertised CLI commands.
- Isolated Python imports resolved from this repository.
- CPU checkpoint hash, schema, and state-shape validation.
- Selected Level 1 configuration hash validation.
- Video hashes/metadata and the report's 10-page count.
- Source-copy hash comparisons and exclusion/path checks performed during preparation.

The checks used the existing Windows environment. A fresh-machine installation, native compilation/loading, new emulator evaluation, and training were not performed. Runtime code and selected models were unchanged by the documentation cleanup.

## Reproduction requirements

Gameplay and training require a legally obtained ROM, compatible native binaries, and the pinned reset/data artifacts. These inputs are not included or publicly hosted. The original owner can restore exact local inputs using the hash-checked asset tool; see [setup](SETUP.md).

Checkpoint loading is verified. End-to-end evaluation and training are not ready from a public clone alone. New DQfD training additionally requires a fresh valid budget/code/protocol chain. A newly compiled native source tree has a different fingerprint from the historical tree; selected-state compatibility needs separate validation.

## Scientific and submission limits

- Known-route clears do not establish seven successful episodes out of ten.
- Level 1 normal-start compliance remains unaudited.
- Imitation agreement on the training route is not held-out accuracy.
- Episode reward telemetry is unavailable for the selected Level 2/3 videos.
- The report is a draft; a revised LaTeX report is not included.
- The short video uses accelerated playback and separate resets/models; acceptance of that presentation format is unresolved.
- The repository is public at the owner's request; the assignment originally specified private access.
- Native-source licensing provenance and project-wide licensing remain unresolved.

## Source and export records

[provenance.json](../manifests/provenance.json) records original/exported hashes; [inventory.json](../manifests/inventory.json) describes current files. [dependency_graph.json](../manifests/dependency_graph.json) and [source_symbols.json](../manifests/source_symbols.json) support implementation review.

Tests, fixtures, private recordings, ROMs, reset states, generated builds, virtual environments, and unrelated research tooling are excluded. The original research workspace preserves them.

To rebuild an export from the original workspace into a new directory:

```powershell
python packaging/prepare_export.py --source <research-checkout> --output <new-empty-path>
```

The exporter refuses an existing destination, follows selected dependency closures, and performs static/help/model checks. It does not launch training or emulator gameplay.
