# Level2 completion verified with a neural imitation policy

The standalone neural actor completed Level2 from its Level2 title checkpoint
in **22509 native frames**, gained **9150 score**,
and finished with **2 lives remaining and15 energy**.
At60 native FPS this is375.15 simulated seconds. The terminal
completion byte is127, with no game-over condition. It is not a short-scene test.

## What the policy uses

The standalone process loads only the checkpoint, the initial full state and
normal emulator/ROM assets. Every action is the greedy output of a559504-
parameter MLP from current2048-byte RAM. It has no frame/record/reset ID input,
reference action sequence, demonstration data, nearest-neighbour table or
state-memory fallback. The raw controller vocabulary contains16 masks, including
fire-plus-vertical combinations excluded by the earlier eight-action studies.
No RAM intervention, mid-episode reset, policy update or optimizer runs during
the completion. A separate reviewer recomputed every neural action and every
gameplay transition, checked hashes/resource samples and decoded the video.

## How it was learned

This is a separate **human-supervised behavior-cloning** branch, not an RL-only
extension of the earlier motion controller. The completed human selected route
provided22449 gameplay actions;60 title/entry-transition actions were added.
All22509 source rows are training data.475 gameplay actions unsupported by the
old17-action vocabulary were recovered as exact raw controller masks from the
hash-pinned read-only SQLite source; no approximate remapping was used.
The old actor, memory, schemas, raw data and models remain preserved.

Initial5000-update cloning reached99.5% action agreement but its continuous run
failed after7096 frames with score0. Low-rate hard-example refinement reached
22449/22449 training agreement and its neural-only run completed from the first
verified gameplay anchor. A further8000-update title-inclusive branch completed
from the earlier narrative checkpoint despite one remaining source-label mismatch.
Training accuracy was never substituted for a live completion test.

The title checkpoint carries stale completion flag127 from the preceding stage.
The runtime and reviewer ignore it until a fresh gameplay entry with flag0 is
observed(frame22); only the later natural flag127 counts.
This avoids a false immediate completion on the title screen.

## Verification and limits

The title-inclusive training evaluator and the standalone checkpoint-only runner
produced byte-identical RAM/actions across all22509 frames. This is
a deterministic reproduction check on the same start, not two independent tests
of generalization. The earlier state-memory baseline also completed, but that
memorized-route result is explicitly separate and is not the selected actor.

**Verified scope: Level2 completion on this recording-derived start/route.**
No robustness rate, unseen-start/new-seed performance, independent-recording
generalization or RL-only learning claim is established. Earlier reserved drop
tests remain separate; this branch does not preserve their previous selection
splits because the entire human route is now supervision. No global model
promotion or replacement of old retention references was made.

Across v54-v58: new native frames117125, uncertainty0, optimizer steps20400.
All executed episodes and failed approaches are retained. The v58 standalone
completion itself has0 optimizer steps and no demonstration/retrieval reads.

## Artifacts and reproduction

- Video: `analysis/level2_standalone_completion_v58_20261026/completion.mp4`
- Policy: `training/level2_title_bc_v57_20261026/actor.pt`
- Start: `training/level2_title_bc_v57_20261026/entry.state`
- Hash-pinned selection: `selected_policy.json`
- Full RAM/actions: `episode.npz`; verification: `verification.json`

Run from the project root, with a fresh unused output path:

```powershell
.venv-gpu/Scripts/python.exe run.py evaluate_level2_completed_actor --checkpoint training/level2_title_bc_v57_20261026/actor.pt --state training/level2_title_bc_v57_20261026/entry.state --output analysis/level2_completion_fresh_run
```

Review that fresh run:

```powershell
.venv-gpu/Scripts/python.exe run.py review_level2_completed_actor --output analysis/level2_completion_fresh_run
```

Existing outputs are occupied and must not be rerun. The next research objective
would be robustness to perturbed actions or independent starts; it is distinct
from the achieved known-route completion.
