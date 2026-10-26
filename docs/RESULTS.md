# Results and evidence

All frame counts below are native emulator frames. Recorded outcomes use separate level models and initial states.

## Level 1: selected DQfD checkpoint

| Global optimizer update | Episode frames | Configured return | Terminal |
| --- | ---: | ---: | --- |
| 1,000 | 36,000 | -22.963344 | Time limit |
| 2,000 | 36,000 | -14.202617 | Time limit |
| **3,000** | **18,273** | **108.091791** | **Level completed** |
| 4,000 | 18,273 | 108.091791 | Level completed |
| 5,000 | 27,246 | 6.722021 | Game over |

Updates 3,000 and 4,000 tied; the earlier checkpoint was selected. All continuation updates used the same 21,599 human/policy demonstration transitions and unchanged reward coefficients. Completion cleared objective IDs 513/514/515/516. Update 5,000 regressed; the cause was not established.

Each checkpoint had one greedy episode from reset 389. These outcomes are not a success-rate estimate or a normal-game-start audit.

- [Selected checkpoint and metadata](../analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json)
- [Continuation results](../analysis/level1_dqfd_demo_continuation_20261025/result.json)
- [Earlier 1,000/2,000 study](../analysis/level1_dqfd_extended_study_20261025/result.json)
- [Training metrics](../analysis/level1_dqfd_demo_continuation_20261025/training_metrics.csv) and [source provenance](../manifests/log_provenance.json)
- [Checkpoint outcomes](../analysis/level1_dqfd_demo_continuation_20261025/candidate_comparison.png) and [learning diagnostics](../analysis/level1_dqfd_demo_continuation_20261025/learning_trends.png)

Loss diagnostics use optimizer updates and are not episode-reward curves. Lower fitting loss did not prevent the final checkpoint's regression.

## Level 2: title-inclusive imitation

The selected v57 neural actor controlled 22,509 title/gameplay frames. An extended recording retained the following evidence:

| Event | Frame |
| --- | ---: |
| Candidate flag | 22,509 |
| Visible Level 2 completion card | 22,809 |
| Level 3 title | 23,409 |
| Level 3 gameplay image | 23,709 |

All pre-flag neural actions and post-flag confirm inputs were independently recomputed. Fresh gameplay entry is required before a new completion flag is accepted. The completion card and next-level title corroborate the flag.

The route gained 9,150 game-score points, collected seven E and three T bonuses, and missed ten of twenty observed opportunities. Game score is not a configured episode return. Exact T-counter time units were not inferred.

The title-inclusive route is training data; deterministic replay from the same start is not independent generalization evidence. See [verification](../analysis/level2_transition_tail_v59_20261026/verification.json), [completion card](../analysis/level2_transition_tail_v59_20261026/after_flag_0300.png), and [next-level title](../analysis/level2_transition_tail_v59_20261026/after_flag_0900.png).

## Level 3: raw-RAM imitation

The 49,100-row selected-ancestry dataset was used entirely for training. The initial 20,000-update fit matched 49,036 source actions; 15,500 further low-rate hard-example updates reached 49,100/49,100 agreement in the single-row runtime inference path. This is training agreement, not held-out accuracy.

The reviewed run retained 49,775 model decisions. The completion card appeared at frame 47,975 and the Level 4 title at 49,175. Every action was independently recomputed. The first 49,100 source-range RAM frames matched the reference; the 675-frame transition tail is not evidence of broad generalization.

There were no mid-run resets, RAM interventions, reference-action substitutions, or optimizer updates. The title comparison permits exact image matching within a bounded vertical scroll offset. Reaching the Level 4 title does not establish Level 4 competence.

See [verification](../analysis/level3_imitation_evaluation_v2_20261026/verification.json), [completion card](../analysis/level3_imitation_evaluation_v2_20261026/frame_047975.png), [Level 4 title](../analysis/level3_imitation_evaluation_v2_20261026/frame_049175.png), and the [initial](../training/level3_imitation_v1_20261026/fit.jsonl) and [refined](../training/level3_imitation_refine_v2_20261026/fit.jsonl) fitting logs.

## Scope

The assignment requires at least seven successful episodes out of ten for each level, with Level 1 starting from the game's normal start. Those criteria remain unestablished. Levels 2 and 3 use imitation learning; the selected Level 1 continuation uses demonstration-assisted TD learning without online interaction. Controlled diagnostic probes are separate from normal-input gameplay evaluation.

## Case study

The [revised case study](case_study/report.pdf) explains the implementation decisions and experimental results in ten main pages plus technical appendices. [Source and build instructions](case_study/README.md) are included.
