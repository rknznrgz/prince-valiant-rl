# User-authorized DQN demonstration-study extension

The user explicitly authorized more time and requested that previous successful
human/policy recordings train the DQN. This is a fresh DQfD demonstration phase,
not a resume of the failed attempt or a rewrite of its outcome.

## Effective allocation

| Limit | Historical pilot | Explicit extension |
| --- | ---: | ---: |
| Cumulative active-time ceiling | 5400 s | 12600 s (+7200 s) |
| Model-I/O cap for this slice | 600 s | 1800 s |
| Demo/encoding cap for this slice | 900 s | 1800 s |
| Whole supervised slice | 1500 s | 3600 s |
| Fresh demonstration updates | historical attempt stopped at1947 | 2000 new updates |
| Cumulative optimizer ceiling | 2160 | 4160 |
| Whole native-frame ceiling | 684640 | 792640 |
| Retained storage cap | 12 GiB | 12 GiB, reserves preserved |

Historical time use remains 3207.6190676 seconds, without reclaiming unknown
history. The new receipt additionally charges a conservative1800 seconds for
amendment engineering/acceptance through initial launch; this is not a measured
duration. The validated prelaunch remaining balance is7592.3809324 seconds.
Whole measured production time is charged once by execute_study.py. Original
historical receipts are unchanged. The final postrun balance is saved separately.

The code interface retains `original_total_active_seconds` as its effective
cumulative-cap field; v2 separately records legacy_original_total_active_seconds
5400 to make the amendment explicit. V1 receipts still enforce the old exact
limits. There is no arbitrary wall-cap override or implicit budget default.

## What trains the model

The existing pinned, read-only reconstruction-v4 transition cache has21599 rows:

| Source | Rows | Role |
| --- | ---: | --- |
| original_389 | 18273 | Verified original human route |
| human18000_noop2 | 836 | Human correction after a two-frame omission |
| human17940_noop2 | 837 | Human correction after a two-frame omission |
| ppo_noop1 | 838 | Verified successful policy recovery |
| ppo_noop16 | 815 | Verified successful policy recovery |

Context/omission frames are not imitation labels. Row proportions are not
measured sampling fractions. All paired one-step/ten-step windows, history
masks, action/controller/native timing and configured rewards stay unchanged.
No dataset copies, relabeling or new human demonstrations were made.

The fresh learner transfers the pinned actor's encoder, then creates new
17-action dueling Q heads, target, Adam, RNG and protected demo priorities. It
does not load checkpoint1000 from the closed attempt. Batch128, gamma0.99997,
beta0.6, paired TD1/TD10/margin objective, Adam1e-4 and target copies at1000/2000
remain unchanged. Greedy reset389 videos occur at1000 and2000. Update2000 is
also the fixed retention gate. At most one further selected1000 demo is allowed
if that candidate remains best, for a maximum three episodes/108000 frames.

## Checks and execution

The public native-free preflight and actual parser dry-run passed. The expanded
production code aggregate is recorded in validated_launch.json; the old
6b2233... hash remains historical because the budget implementation changed.
No learner/inference/resource/natural-terminal behavior was changed to pass a gate.

Focused acceptance: five new extension tests, five legacy budget propagation
tests, four legacy loop/serialization tests and one retention-ranking test passed.
New extension tests use fake update events/state facades, zero real model
forwards/optimizer steps/native loads/CUDA initialization. They exercise exact
accounting, invalid links/ticks, bounded caps, the actual2000-event loop beyond
600 seconds with both demos, and public-parent/private-child budget propagation.

The separately invoked older fault matrix initially ran34 methods and reported18
errors, including subtest errors: stale fixtures serialized MappingProxyType or
omitted the already-mandatory execution_budget receipt. The test fixtures were
then repaired to supply explicit hash-linked receipts, complete current-schema
fake resource evidence, and the existing typed failure reasons. Deadline tests
now bind to durable publication rather than a brittle count of clock calls.
All34 fault-matrix tests passed after migration, for49 focused tests overall.
Production validation was not weakened, and the running production implementation
was unchanged. The migration backup and both failed intermediate outputs are
retained alongside the final passing output. See verification.json.

execute_study.py invokes one preflight, one dry-run, and at most one actual
public pretrain command. Original stdout and argv are retained. The existing
supervisor performs fresh launch/runtime resource sampling and owned-child
cleanup. No standalone resource polling, broad process termination, installation
or native build is performed. No retry follows a runtime/resource/provenance
failure. No online/B/C/final episode automatically follows this slice.

Source changes: dqfd_study_budget.py, receipt dispatch/caps in
dqfd_pretrain_entrypoint.py, amended loop-cap validation and dynamic limit error
messages in dqfd_pretraining.py. Legacy protocol files and all original datasets,
receipts, checkpoints and videos are preserved. New preparation CLI and fake-only
tests make the amendment reviewable. No root Git metadata exists, so unrelated
concurrent edits cannot be enumerated.

## Completed results

The fresh slice completed2000 attempted/applied/successful updates,2000 durable
diagnostics and priority updates, two target copies and both checkpoints. The
parent accepted the complete saved-state/receipt chain and the owned processes
exited cleanly. This is successful execution/persistence, not a passed policy
retention gate. All2169 parent resource samples passed.

| Candidate | Native frames | Configured reward | End | Verified objective clears |
| --- | ---: | ---: | --- | ---: |
| [1000 video](../../training/level1_dqfd_demonstration_extension_20261025/videos/milestone_1000_candidate.mp4) | 36000 | -22.96334355391665 | time_limit | 0 |
| [2000 video](../../training/level1_dqfd_demonstration_extension_20261025/videos/milestone_2000_fixed2000_gate_candidate.mp4) | 36000 | -14.202616936275213 | time_limit | 0 |

Update2000 is best among these two observed candidates under the preserved
ranking, but its fixed2000 retention gate **FAILED**. No extra selected1000
episode was needed. No online/B/C/final episode followed. The old attempt's
fixed2000 gate remains historically NOT_EVALUATED; the failed measurement above
belongs to this new study. Neither two milestones nor one seed establish a
robustness rate or assignment7/10 acceptance.

Whole public-command interval1188.2035239s. The final phase-complete receipt
records669.4722458s model-I/O including setup/cleanup and515.1915822s demos/encoding,
for1184.663828s in that child-accounted interval. These are wall-accounting
segments, not GPU kernel timings. The larger cap resolved the old deadline
problem: the actual model-I/O interval exceeded the old600s allowance and stayed
below1800s. The small difference from the whole public-command interval is not
further attributed without substage measurements.

The final conservative cumulative balance is61958225915 used ticks and
64041774085 remaining ticks, i.e.6195.8225915s used /6404.1774085s remaining under
12600s. It preserves all3207.6190676s of historical charge and the new1800s
conservative preparation debit. New native work was72000 frames, zero uncertain,
two episodes: cumulative153014/792640, remaining639626. Cumulative optimizer
usage is4107/4160 steps (53 remain as accounting), durable diagnostics4106 and
counter-based presentations525696. Remaining balances do not authorize another
automatic run, and the failed fixed2000 gate still blocks online handoff.

The first/last100-update total-loss medians were0.622225/0.191683. TD1 median
declined0.146519 to0.095135; TD10 median rose0.204206 to0.389254. See
[learning_trends.png](learning_trends.png). Declining fitting loss is not a
completion result. Sampled source fractions and unique rows remain NOT_MEASURED.

All five Q/target/Adam/RNG/PER state hashes at1000 match the earlier attempt's
1000 receipt. Its action/video hashes also match. The checkpoint binary hash
differs because its protocol/code/budget metadata records this new study.

The [2000 contact sheet](update2000_contact_sheet.png) contains12 requested saved
timestamps from0 to600s, with approximate native-frame mapping in video_review.json.
It shows different grassy/bridge positions early, then repeated similar bridge
placement in the last six samples (327.273–600s). This is consistent with a stall;
the sparse samples do not establish the exact action sequence or continuous
motion. The video was stream-hashed against its committed episode receipt; no
new gameplay was generated for the review.

**Next learning priority:** diagnose the bridge stall against the verified human
trajectory before choosing another training change. More runtime solved execution
feasibility; it has not yet solved greedy control. No algorithm/precision/batch/
reward or sampling retuning was performed in this budget-only study.

`decision.json` is the preserved preparation-time decision. `result.json`,
`verification.json`, `postrun_budget.json` and the final artifact ledger describe
the completed outcome. The source/hash/protocol/receipt chain and failed test
migration outputs remain available for review.
