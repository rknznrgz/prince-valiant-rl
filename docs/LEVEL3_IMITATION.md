# Level3 human-imitation task

This is a separate Level3 observation/model contract. The goal is a normal-input
neural policy that finishes the recorded Level3 route; robust generalization is
not required or implied. Human action supervision is explicitly authorized.

## Observation contract

`training/level3_imitation.py` defines `level3_raw_ram_imitation_v1`: an independent
copy of the current native2048-byte `uint8` RAM observation. All bytes are opaque
numeric inputs. No Level2 enemy, hitbox, resource, score or terminal semantics are
transferred. All bytes are available; invalid dtype/shape fails closed. Training
normalization is per byte, scaled by255, with standard deviation floored at0.02.
The model consumes neither source record IDs nor demonstration frame numbers,
future RAM, reset identity, action-table matches or a sequence cursor. Internal
game counters remain observable RAM, not injected external progress labels.

The actor is an MLP2048→512→256→22 raw controller-mask logits. It is initialized
separately; Level1/2 checkpoints and feature shapes remain unchanged. Source masks
are preserved exactly rather than remapped into Level2's restricted action set.
Current RAM uniquely distinguishes all49100 source action rows. A future dataset
with conflicting same-observation labels would require a new history contract.

## Demonstration and reset provenance

`analysis/level3_imitation_v1_20261026_preparation/` contains the read-only,
hash-pinned selected-ancestry extraction for path frames44700..93800. It begins
in Level3's intro before confirmed gameplay and includes the completion/Level4
title transition. The source recording contains rewinds; extraction follows its
chosen ancestry with exact post/pre RAM continuity. Execution must load one
initial full state and then run uninterrupted, with no mid-run rewinds or resets.
All49100 rows are training data. No held-out accuracy/generalization claim applies.

`analysis/level3_observation_validation_v1_20261026/` verifies360 native transitions
at three checkpoints, plus rendered terminal references. Loading a checkpoint's
screen alone can expose stale composition; screenshot acceptance uses active
emulator rendering. `analysis/level3_terminal_references_v2_20261026/` actively
renders the Level3 completion card and Level4 title for exact text-region checks.

## Completion acceptance

The measured F3 transition in this source route is candidate evidence only.
The intro carries a stale127; gameplay entry must be observed first. A successful
evaluation must retain the visible **LEVEL3 COMPLETE** card and following
**Level4 – The Sea Battle** title. A byte flag, timeout horizon or training fit
alone cannot establish completion. Inspect the ending screenshots/video and run
the independent all-action reviewer.

## Tools

From the project root, use fresh output paths; occupied artifacts are preserved.

```powershell
.venv-gpu/Scripts/python.exe run.py train_level3_imitation --output training/level3_fresh
.venv-gpu/Scripts/python.exe run.py evaluate_level3_imitation --checkpoint training/level3_fresh/best.pt --output analysis/level3_fresh_run
.venv-gpu/Scripts/python.exe run.py review_level3_imitation --output analysis/level3_fresh_run
```

The inference runner loads only a checkpoint, reset state, ROM/native library
and small contract metadata. It does not load demonstration RAM/actions. Any
post-Level3 model actions are retained for transition evidence, not claimed as
Level4 gameplay competence. All actions must be ordinary controller masks; RAM
interventions, reference playback, action retrieval and reset-assisted stitching
are excluded from the completion test.
