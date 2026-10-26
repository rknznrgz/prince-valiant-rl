# Real Level2 completion and bonus-collection audit

**The extended run confirms a real clear, not a timeout.** The earlier video ended at the flag change while the forest screen was still visible. It did not show the corroborating transition, which made the completion claim insufficiently presented. This run continues through the completion card, Level3 title, and Level3 gameplay.

|Evidence|Native frame from Level2 title start|Artifact|
|---|---:|---|
|Candidate flag|22509|after_flag_0000.png|
|Visible LEVEL 2 COMPLETE|22809|[Completion card](after_flag_0300.png)|
|Visible Level 3 – The Viking|23409|[Next-level title](after_flag_0900.png)|
|Visible Level3 gameplay|23709|[Next-level gameplay](after_flag_1200.png)|

[Full run including transition](full_transition.mp4). The ordinary confirm-button presses after the completion card are recorded; there are no resets or RAM edits. The learned actor controls the22509-frame title/gameplay segment, then explicit menu-confirm inputs advance the post-clear screens. No Level3 gameplay competence is claimed.

## E/T collection evidence

Before the clear, the run has20 observed kind8 bonus lifetimes:10 collected,10 missed. Seven collected phase2 E bonuses each add128 raw energy (896 total). Three collected phase1 T bonuses change the ROM-bound timer/resource counters. No unit conversion of those raw counters is assumed. Every event requires guard-enabled8->9 plus the game's100-point award; mere alignment, firing or disappearance is not counted.

|Bonus|Native frame|Slot|Energy gain|Raw counter before → after|
|---|---:|---:|---:|---|
|E|4801|1|128|[0, 0, 1] → [0, 0, 1]|
|E|13232|3|128|[0, 1, 0] → [0, 1, 0]|
|T|13724|1|0|[0, 0, 5] → [0, 1, 3]|
|E|14460|0|128|[0, 1, 2] → [0, 1, 2]|
|E|15272|1|128|[0, 1, 1] → [0, 1, 1]|
|E|16802|1|128|[0, 0, 4] → [0, 0, 4]|
|T|17300|0|0|[0, 0, 3] → [0, 1, 1]|
|T|18319|3|0|[0, 1, 0] → [0, 1, 3]|
|E|18580|0|128|[0, 1, 2] → [0, 1, 2]|
|E|20089|0|128|[0, 1, 0] → [0, 1, 0]|

The model therefore does shoot and collect E/T bonuses, but misses half the observed opportunities. Completion does not establish optimal collection. The earlier report should have quantified these pickups and shown the next-level transition instead of relying on the flag/video cutoff.

## Improvement attempts

Two fresh normal-input variants prioritized current guard-enabled bonuses: one used the trained motion pickup controller, the other used centre aiming with fire-plus-direction. Both disrupted the learned trajectory, collected0 bonuses and ended in game over at7816/7984 frames. They were rejected; no improvement in collection rate is claimed. See [negative comparison](../level2_bonus_priority_v60_20261026/README.md). The unchanged completion-capable policy is retained.

## Verification and scope

All neural actions through the candidate flag and all post-flag confirm inputs were independently recomputed. The prefix matches the prior standalone RAM/actions exactly. The completion-card text region matches an independently retained human completion witness. The Level3 title and subsequent Viking gameplay screenshots were visually inspected. The new gate requires fresh gameplay entry, a candidate flag, the completion card, then the next-level title; flag alone is rejected. Four focused tests cover stale flags, incomplete evidence, blank templates and guarded bonus accounting.

Extended run: 24909 native frames, uncertainty0, optimizer0; full video decoded and85 resource samples pass. Priority experiments: 15800 frames, uncertainty0, optimizer0. Their negative terminals/bonus outcomes were reclassified from saved RAM and videos decoded; no gain is claimed from those variants.

This remains a human-supervised policy on the training recording-derived start. It is not evidence of RL-only learning, independent-start robustness, optimal E/T collection or Level3 completion. All original checkpoints and historical receipts remain preserved; this stronger audit supersedes flag-only interpretations.

Fresh extended evaluation, using an unused output path:

```powershell
.venv-gpu/Scripts/python.exe run.py evaluate_level2_confirmed_completion --output analysis/level2_clear_fresh
```

That runner records the complete transition and writes completion_evidence.json from card/next-title templates; inspect its screenshots and video before asserting a clear. Existing output directories must not be reused.

## Sparse-fire follow-up

Two additional variants preserved directional bits and only added sparse fire for any eligible bonus or T bonuses. Both failed (8489/8247 frames,0 pickups) and were rejected. See [verified sparse-fire comparison](../level2_bonus_firefill_v61_20261026/README.md). No collection improvement is claimed across these four rejected variants. The actual Level2-clear proof remains the unchanged original policy.
