# Additional demonstration training from committed update2000

User authorization: train further to measure improvement; reward changes and
coefficient experiments permitted. This block first holds rewards fixed to
isolate additional training. The original cache stores per-transition rewards
under pinned weights: changing evaluation coefficients alone would not train a
new reward objective. A later reward experiment needs consistently relabeled
paired returns or a new verified cache, original cache retained, with evaluation
under matching coefficients. No coefficient changes were made in this block.

## Exact continuation

Continue from the completed, parent-accepted update2000 checkpoint of
`training/level1_dqfd_demonstration_extension_20261025/`. Verify its exact binary
hash, committed receipt, metadata, counters and full Q/target/Adam/RNG/PER hashes.
Restore every learning state before updating; verify hashes again after restore.
The earlier failed1947 attempt is not resumed. The previous completed2000 phase
had a failed performance gate, but valid saved-state integrity; user authorization
now permits further offline demonstration training. This does not bypass an
online-collector gate or relabel the previous fixed2000 result.

Train global updates2001–5000 from the same21599 human/policy demonstration
transitions. Greedy reset389 checkpoints/videos at3000/4000/5000, no extra best
episode, maximum3 episodes108000 native frames. Constant beta0.6,128-row batches,
Adam1e-4, paired TD1/TD10 and target copies every1000 remain unchanged. Checkpoint
loader already supports global milestones and exact target-copy counts.

## Allocation and execution

The cumulative time ceiling remains12600s. Preserve prior6195.8225915s and add
a conservative600s preparation debit:5804.1774085s remain before this command.
This is an engineering allowance, not a measured duration. Model-I/O and demo
segments each1800s, whole supervised slice3600s. Add3000 update allowance and
108000 native frames: cumulative ceilings7160 applied steps and900640 frames.
Existing12GiB retained cap and reserves remain. Historical53 unused steps are
accounting only, not a substitute for this explicit continuation authorization.

The same public CLI/parent supervisor, ownership/resource gates and child receipt
chain run this block. A separate v3 receipt binds the user authority and the
preceding final ledger. No automatic retry, online training, B/C or final episode.
`execute_study.py` runs preflight, dry-run and one production command. Every
original checkpoint/report/cache/video remains unchanged; all new output uses a
fresh directory. Root Git metadata is absent; concurrent edits cannot be listed.

## Verification

Five new fake-only tests cover exact state restoration/hash corruption, forged
receipts, wrong start/deadline rejection,3000 update events/three demos and an
invalid demo stopping before3001. They run no optimizer/forward/native work.
Five legacy budget tests, five prior extension tests and34 fault-matrix tests
also passed. Compile checks passed. Real preflight/dry-run and the supervised
result are recorded separately; a dry-run does not imply completed training.

The new continuation module is in the production code hash. Factory/loop
selection occurs only for the validated explicit continuation study ID. Legacy
fresh pretraining still rejects nonzero start counters. Historical fixed2000
remains failed; additional milestone outcomes must be reported separately.

## Completed results: Level1 clears at3000 and4000

All3000 additional attempted/applied/successful updates through global5000,
3000 diagnostics/priority updates and three new target copies are durable.
Checkpoints/videos at3000/4000/5000 are committed. The final saved-state/receipt
chain was parent accepted, owned processes exited cleanly, and all2482 resource
samples passed. Execution success is separate from candidate performance.

| Global update | Native frames | Configured reward | Recorded terminal | Objective IDs |
| --- | ---: | ---: | --- | --- |
| 2000 baseline | 36000 | -14.202616936275213 | time_limit | none |
| [3000 video](../../training/level1_dqfd_demo_continuation_20261025/videos/milestone_3000_continuation_candidate.mp4) | 18273 | 108.09179112744023 | natural level_completed | 513/514/515/516 |
| [4000 video](../../training/level1_dqfd_demo_continuation_20261025/videos/milestone_4000_continuation_candidate.mp4) | 18273 | 108.09179112744023 | natural level_completed | 513/514/515/516 |
| [5000 video](../../training/level1_dqfd_demo_continuation_20261025/videos/milestone_5000_continuation_candidate.mp4) | 27246 | 6.722020514690886 | natural game_over | 513 |

**Selected observed candidate: update3000**, tied with4000 and selected by the
earlier-update tie rule. `selected_checkpoint.json` records its exact reference.
It is preserved in place, never overwritten by5000. More training improved the
measured policy through3000/4000 and then regressed at5000. These episodes do not
establish the mechanism causing regression; overfitting/instability are unproven.

This is one greedy episode per checkpoint from verified reset389, not an
independent7/10 acceptance study. No condition-B/C robustness, normal-game-start
compliance or Levels2/3 competence is inferred. Historical fixed2000 remains
FAILED. The legacy phase status `pretraining_complete_retention_failed` preserves
that historical gate while the new3000/4000 completions are recorded separately.
No online phase followed. The3000/4000 video hashes match. The4000 action/mask/
duration trace matches the recorded human route;3000's differs despite identical
video/reward/length. The production callback executed greedy Q actions with no
reference-action substitution or new human input.

Whole command1353.6227288s (22m34s); final child receipt model-I/O940.2470050s
including setup/cleanup, demos/encoding409.9115938s. These are wall segments,
not GPU kernel measurements. Final cumulative used81494453203 ticks
(8149.4453203s), remaining44505546797 ticks (4450.5546797s) under12600s, including
the conservative600s preparation debit. Historical unknown time was not reclaimed.
New native work63792 frames, three episodes, zero uncertain; cumulative216806/
900640 frames. Cumulative optimizer7107/7160, diagnostics7106, presentations909696.
Remaining balances do not authorize automatic work.

No nonfinite scalar diagnostics. First/last100-update loss medians0.194360 to
0.163278; TD1 medians0.091070 to0.078578, TD10 medians0.392851 to0.354618. Fitting
loss decline did not prevent5000's performance regression. See
[candidate comparison](candidate_comparison.png) and [learning trends](learning_trends.png).
Unique sampled rows/source fractions remain NOT_MEASURED; all original replay
and reward coefficients are unchanged.

Eight saved frames per video (24 total) were selected at recorded requested times.
The3000 samples show varied grassy/bridge and stepping-stone scenes; the5000
late samples repeatedly show similar bridge compositions, with no avatar visible
in the last selection. Sparse samples cannot establish exact action repetition
or the cause of death. Natural-terminal receipts remain authoritative.

**Next recommendation:** hold training at the selected3000 candidate and audit
normal-start/provenance and acceptance evaluation before more coefficient
experiments. There is now a successful Level1 candidate to validate. Do not
prefer5000 merely because it is later. No online/B/C/final episode automatically
ran. Changed reward coefficients would require consistent replay-target handling.
