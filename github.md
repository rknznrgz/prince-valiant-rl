# GitHub preparation and report implementation review

Prepared **26 October 2026**, local package `github_submission/prince-valiant-rl`
beside original `prince-valiant-rl`. No research-root Git metadata exists: no
revision identity, clean diff or commit history is claimed. Original tests,
experiments, recordings and models are preserved. Only the research root github.md
and status handoff are updated. No publication/upload/access change occurred.

This file is the concrete retained-code handoff for the research
`docs/REPORT_PREPARATION_PROMPT.md`. The file/function map below describes the
staging implementation. [Provenance](manifests/provenance.json) records original
and exported SHA256 values; do not confuse adaptation hashes with historical
experiment code identities.

## Supported implementation and readiness

| Workflow | Included functionality | Current readiness |
|---|---|---|
| Level 1 DQN/DQfD | Exact update-3000 checkpoint, committed receipt, DuelingQ, losses/PER, transition builder, supervised trainer and exact continuation loader, reset-389 evaluator | CPU checkpoint load verified; native evaluation blocked on route/states, identity-pinned DLL/provenance and ROM; training additionally needs pinned replay/actor/budget chain |
| Level 1 PPO | Backward PPO trainer/evaluator, process workers, curricula, rewards, temporal models | Source/help checked; route, normalization/BC/config-specific inputs required; no selected PPO result replaced by this package |
| Level 2 imitation | Title-inclusive fitting from v56 parent; neural completion runner and independent visual card/title check | Model load verified; exact data/parent/reset and native inputs withheld; not training/evaluation ready as distributed |
| Level 3 imitation | Raw-RAM fitting/refinement, selected actor, current-RAM evaluator and independent all-action/reference review | Model load verified; normalization embedded; exact preparation/dataset/reset/reference images needed for evaluation/review |
| General environment | Gymnasium RAM wrapper, ROM identity, extraction, actions and state loading | Source/help checked; nes-py wheel and gameplay state required; legacy wrapper differs from selected RecorderNative workflows |

Exact commands are in [README](README.md), with safe `--help` captures and results
in [verification.json](manifests/verification.json). Implemented means retained
source; static/help checks do not mean runtime gameplay. Only CPU checkpoint
deserialization/state shape and integrity checks were executed. No optimizer,
training, native build/load, emulator gameplay or fresh evaluation was run.

## Dependency and required artifact map

| Workflow | Code closure and artifact/schema relationships |
|---|---|
| Native/environment | recorder_native.RecorderNative -> recorder_api.cpp -> nes_emu include/src; ROM identity and build-bound PVSTATE1 reset payload. kind18_observer.Kind18ObserverNative adds ABI2 sidecar and provenance. rl_env.PrinceValiantRamEnv is a separate legacy nes_py wrapper requiring gameplay.json. |
| Level1 | prepare_level1_dqfd_transitions -> lazy dqfd_level1_adapter:create_adapter -> Level1Env/HumanRoute -> route.json/route.npz/states; TransitionCache -> manifest/READY/identity/outcomes/transitions/windows. DuelingQ -> WindowPath -> schema/shapes/model_config inside committed checkpoint. dqfd_execution -> paired TD1/TD10, margin, PER; continuation restores Q/target/Adam/RNG/PER only from accepted update2000. Selected eval uses committed update3000 + metadata pins + original reset389 identity. |
| PPO | train_backward_level1 -> process_envs -> backward_level1.level1_child; TemporalActorCritic/FrameHistory/window_storage/control_history -> same versioned feature groups, curriculum/reward configuration. Selected score-quarantine config retained for DQfD lineage; base backward config is an additional supported trainer example, not a reproduction of every report variant. |
| Level2 | prepare_level2_replay -> selected ancestry/SQLite -> RAM replay; title preparer -> index checkpoint20700/title.state/intro.npz; run_level2_title_bc -> v4 replay + intro + SQLite labels + v56 parent -> DemonstrationActor schema level2_full_ram_bc_v55, normalization buffers inside actor.pt. Evaluation -> frozen v57 actor + entry.state + original authority; transition tail uses ordinary confirm-button presses after flag, not neural Level3 play. CompletionEvidence requires complete card then Level3 title. |
| Level3 | prepare_level3_imitation -> immutable SQLite + segmentation/index -> 49100 rows44700..93800 + frame44700 reset + preparation.json; Level3ImitationActor schema level3_raw_ram_imitation_v1 -> 2048/512/256/22, mean/scale embedded. Refinement uses same vocabulary, starts fresh Adam. evaluate -> checkpoint/preparation/reset/ROM/DLL only; review -> full saved actions/RAM + training dataset + actively rendered independent terminal references. |

Full transitive edges and explicit dynamic dispatch/source dependencies:
[dependency graph](manifests/dependency_graph.json). Missing inputs have actual
size/hash/destination and owner-restoration instructions in
[artifacts](manifests/artifacts.json). No alternative recordings are asserted
hash-compatible. Where acquisition is owner-only, public reproduction is blocked.
Historical budget receipts bind old source hashes; staged code changes invalidate
reuse for a new DQfD training launch. A fresh reviewed budget/code/protocol chain is
required, not a copied receipt or bypassed guard. This is a training-readiness gap.

## Assignment evidence and gaps

| Assignment requirement (direct DOCX read, including tables) | Evidence | Status/gap |
|---|---|---|
| Complete Levels 1–3 in at least 7/10 deterministic/low-temperature episodes | Recorded known-route clears | NOT established; no 10-episode campaign |
| Level 1 evaluation from normal game start; no lives/invincibility/level-skip manipulation | Selected reset-389 study | Normal-start compliance unaudited; do not count controlled probes |
| Separate level models allowed; documented imitation/curricula allowed | Separate DQfD / Level2 and Level3 imitation | Disclosed; not one generalizing model, not all online RL |
| Environment, training, evaluation source | Curated dependency graph and source map | Included; native/data blockers below |
| Setup/commands | README and CLI help captures | Prepared; no clean-machine run |
| Pinned dependencies and seeds | requirements files; experiment/config/model metadata | Installed versions verified; package-index availability unverified; L1 DQfD 20261024, base PPO config 20261015 (selected lineage 20261022), L2 20261128, L3 20261130; see exact retained config for each branch |
| Trained checkpoints/download | Three selected models, receipt and artifact manifest | Included; no public data downloads |
| Logs/TensorBoard | Authentic L2/L3 fit.jsonl, results; L1 learning plots and original diagnostics ledger listed below | Prepared with units; not all requested episode curves available |
| PDF 5–10 pages, optional appendices | REPORT_DRAFT.pdf (10 pages) | Draft only; revised LaTeX replacement pending |
| Report algorithm/reward/exploration/hurdles/results/time/hardware/next steps | Draft report and retained configs/logs | Draft review required; no invented training time or missing curves |
| Stable 3–5 minute 720p+ MP4, early/middle/final/full runs/steps/rewards | Four hash-verified videos | 263.67s presentation; accelerated, separate runs; Level2/3 reward N/A |
| Private Git repository with Alphabots access and one submission link | No publication attempted | Future step; access NOT granted |
| Due date | Assignment prose says 13.10.2026 (change); table says 16.10.2026 | Conflicting source dates, no deadline inferred |
| Bonus comparisons/ablation/Level4/generalization | Scoped PPO/DQfD history; no controlled reward ablation included | No extra-credit acceptance claimed; Level4 title is not competence |

## Checkpoints, logs, report and videos

Level1: `training/level1_dqfd_demo_continuation_20261025/checkpoint_3000.pt`, SHA256
`7e44f1bc067b8cb827dfa8594a15fe7a7a51896a7bf0ab9ff104977c6f97d7c6`.
Selection and exact configuration/shape/seed/state hashes are retained in
`analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json`.
Updates3000 and4000 completed;5000 regressed. Only3000 is exported. This is
demo-only DQfD continuation, not online RL. Configured episode reward at3000 is
+108.09179112744023 over18273 native frames. No global best/latest substitution.

Level2: `training/level2_title_bc_v57_20261026/actor.pt`, SHA256
`900c098cbe0376cfbc5aabc4c145b83fcd96b377772c6528e1a42beb9e9efbbd`.
Level3: `training/level3_imitation_refine_v2_20261026/best.pt`, SHA256
`1481dd27a35a04e97bcb9dd25717171d16d25fa9063e18cda69e404f9f5afbcb`.
All checkpoint byte sizes/hashes are in artifacts.json; normalization lives in
the two imitation state dictionaries. Vocabulary and schema are preserved.

Authentic `fit.jsonl` logs use optimizer update indices, cross-entropy and fitted
row counts; they are not native-frame reward curves or held-out accuracy. L3
initial and refined fits are separate, total35500 applied supervised updates.
L1 training_metrics.csv contains3000 authentic update rows extracted from the
source receipt journal; manifests/log_provenance.json pins the original journal
and derived table hashes. Learning plots/README show milestone outcomes. Evidence remains scoped
to each original experiment. File inventory lists exactly which logs are included.

Draft10-page report: [PDF](docs/case_study_draft/REPORT_DRAFT.pdf), [text](docs/case_study_draft/REPORT_DRAFT.md).
Video: [presentation](deliverables/videos/progress_presentation.mp4), plus three
full-run real-time companions; actual source/output metadata in
[video manifest](deliverables/videos/manifest.original.json). Export inventory
adds current hashes. Original video manifest hash is tracked separately.
Presentation263.67s,1280x960,30fps H.264;24x early/middle,8x final playback;
no internal source-run cuts, separate resets/policies, no recorded L2/3 reward,
no voiceover/game audio. Re-decoding is not a new gameplay verification.

## Included/excluded and portability

The explicit exported inventory is [inventory.json](manifests/inventory.json).
Only import closure files, narrow native include/src + build wrappers, selected
models/evidence/report/videos and packaging tools are included. Tests, fixtures,
fake harnesses, archive copies, agent/session tooling, private notes, generated
builds, ROMs, recordings, reset states, virtual environments and duplicate models
are excluded; they remain in the source workspace. .gitignore is supplementary.

Portability changes: explicit CLI catalog with no test command; fresh output
parsers for one-shot Level2/title and Level3 preparers; removal of generated-script
rewriting and unused Level2 rollout code; direct Level2 completion runner import;
new strict selected-model inspector, local asset restorer and Level1 evaluator.
Model architectures, reward coefficients, schemas and committed models unchanged.
Original experiment hashes stay in provenance; new execution hashes differ.
The frozen DQfD protocol hash is retained directly instead of distributing private
agent prompts. Validation still requires that exact hash and a valid budget chain.
Native build identity originally hashed the entire copied nes_emu tree, including
unretained native tests/benchmarks. The minimal source build therefore has a new
tree identity: historical reset compatibility requires a separately reviewed codec
identity procedure or access to the exact original locally held binary. A new
build is a build route for new states, not proof of selected-state compatibility.

## Verification and publication blockers

See verification.json for exact subprocess argv, return codes and captured help,
CPU model integrity results, static closure audit and metadata checks. Interpreter
is the author's existing CPython environment, with isolated mode and explicit
staged paths. This is NOT a clean-machine install test. No native smoke/evaluation
check was performed. Native compiler compatibility and published wheel availability
remain unverified. DLL hash pins/build-bound states may reject clean rebuilds.
Private inputs have no public delivery plan. No general Level2/3 environment
semantic validity or unseen-route competence is asserted.

Resolve native license provenance, project license choice, dataset/reset/media
redistribution, fresh DQfD authority chain, input access, fresh-machine dependency
install/native compatibility, final report, normal-start/7-of10 evaluation and
video-format acceptance before claiming a compliant submission. Two video files
exceed50MiB; all are below100MiB, but cumulative history/storage needs a decision.
No LFS pointers, remote repository, release links or granted access are fabricated.

## Scientific limits

One known human-route clear is not a7/10 campaign. Level1 normal-start compliance
remains unaudited. Level2/3 are imitation-based and fitted on the full route,
without held-out generalization. Controlled probes remain separate from normal
game evaluation. Visible Level4 title does not demonstrate Level4 competence.

## Exact retained file/function map

| File | Exact top-level symbols | Retained role |
|---|---|---|
| [scripts/audit_manual_recording.py](scripts/audit_manual_recording.py) | `audit`, `main` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [scripts/evaluate_level2_confirmed_completion.py](scripts/evaluate_level2_confirmed_completion.py) | `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/evaluate_level2_transition_tail.py](scripts/evaluate_level2_transition_tail.py) | `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/evaluate_level3_imitation.py](scripts/evaluate_level3_imitation.py) | `main` | Raw-RAM imitation or independent Level3 evidence |
| [scripts/evaluate_selected_level1.py](scripts/evaluate_selected_level1.py) | `main` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [scripts/inspect_models.py](scripts/inspect_models.py) | `main` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [scripts/level1_dqfd_pilot.py](scripts/level1_dqfd_pilot.py) | `main` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [scripts/level2_v12_rollouts.py](scripts/level2_v12_rollouts.py) | `digest`, `write`, `resource_gate` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/prepare_backward_level1.py](scripts/prepare_backward_level1.py) | `dump`, `recording_identity`, `assert_same_snapshot`, `branch_path`, `main` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [scripts/prepare_level1_dqfd_transitions.py](scripts/prepare_level1_dqfd_transitions.py) | `_write_parent_failure_receipts`, `digest`, `tree_bytes`, `verify_preserved_budget`, `bounded_array`, `preflight`, `main` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [scripts/prepare_level2_replay.py](scripts/prepare_level2_replay.py) | `digest`, `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/prepare_level2_score_replay.py](scripts/prepare_level2_score_replay.py) | `digest`, `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/prepare_level2_title_bc.py](scripts/prepare_level2_title_bc.py) | `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/prepare_level3_imitation.py](scripts/prepare_level3_imitation.py) | `main` | Raw-RAM imitation or independent Level3 evidence |
| [scripts/review_level3_imitation.py](scripts/review_level3_imitation.py) | `crop`, `main` | Raw-RAM imitation or independent Level3 evidence |
| [scripts/run_level2_title_bc.py](scripts/run_level2_title_bc.py) | `main` | Level2 data/actor/reward/completion evidence dependency |
| [scripts/submission_assets.py](scripts/submission_assets.py) | `sha`, `main` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [scripts/train_backward_level1.py](scripts/train_backward_level1.py) | `write_json`, `sha256_file`, `log`, `training_memory_sample`, `training_case`, `apply_fixed_anchor_prefix`, `empty_combat_scope_collection`, `accumulate_combat_scope_transition`, `summarize_combat_scope_collection`, `empty_score_scope_collection`, `accumulate_score_scope_transition`, `summarize_score_scope_collection`, `reset_resumed_worker_state`, `reset_collector_role_worker_state`, `is_frontier_mastery_transition`, `reset_record`, `entropy_weight`, `critic_warmup_rollout`, `training_budget_targets`, `configuration_diff_paths`, `configuration_changed_keys`, `validate_curriculum_adaptation`, `validate_timing_adaptation`, `validate_normalization_adaptation`, `rng_state`, `load_rng`, `validate_reward_adaptation`, `validate_score_reward_adaptation`, `validate_adaptation_composition`, `validate_score_curriculum_adaptation`, `validate_scoped_kind18_reward_resume`, `validate_navigation_reward_adaptation`, `validate_time_cost_adaptation`, `validate_forward_progress_adaptation`, `optional_training_counter`, `validate_evaluation_identifier`, `snapshots`, `adaptation_validation_result`, `validate_checkpoint_experiment_provenance`, `checkpoint_route_observation_window`, `validate_evaluation_checkpoint_contract`, `validate_action_omission_plan`, `action_with_one_shot_omission`, `compare_reference_greedy_action`, `policy_input_schema_matches_reference`, `fingerprint_schema_matches_reference`, `evaluation_preparation_result`, `save`, `behavior_distribution`, `exploration_epsilon`, `task_score`, `validate_bc_mode`, `auxiliary_weight`, `load_human_auxiliary`, `human_auxiliary_loss`, `optimize`, `flush_evaluation_values`, `evaluate`, `curves`, `main` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [scripts/train_level3_imitation.py](scripts/train_level3_imitation.py) | `main` | Raw-RAM imitation or independent Level3 evidence |
| [src/prince_valiant_rl/backward_level1.py](src/prince_valiant_rl/backward_level1.py) | `section0_timeout_terminal`, `TaskHistory`, `HumanRoute`, `Level1Env`, `BackwardCurriculum`, `level1_child` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [src/prince_valiant_rl/control_history.py](src/prince_valiant_rl/control_history.py) | `ControlHistory` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [src/prince_valiant_rl/forward_curriculum.py](src/prince_valiant_rl/forward_curriculum.py) | `load_manifest`, `milestone_reached`, `ForwardExposureCurriculum` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/game_state.py](src/prince_valiant_rl/game_state.py) | `word`, `signed_byte`, `decode_display_score`, `forest_dispatch_valid`, `extract_game_state` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/kind18_events.py](src/prince_valiant_rl/kind18_events.py) | `scene`, `Kind18Events` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/kind18_observer.py](src/prince_valiant_rl/kind18_observer.py) | `Kind18Event`, `Kind18ObserverNative` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/level1_potential.py](src/prince_valiant_rl/level1_potential.py) | `final_encounter_potential`, `potential_difference` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/navigation.py](src/prince_valiant_rl/navigation.py) | `map_section`, `supported`, `live`, `completed`, `NavigationInterventions`, `NavigationHistory`, `NavigationEncoder`, `NavigationEnv`, `navigation_child` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/navigation_native.py](src/prince_valiant_rl/navigation_native.py) | `library`, `Snapshot`, `NavigationNative` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/recorder_native.py](src/prince_valiant_rl/recorder_native.py) | `library`, `configure_library`, `RecorderNative` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/rl_env.py](src/prince_valiant_rl/rl_env.py) | `PrinceValiantRamEnv` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/rope_geometry.py](src/prince_valiant_rl/rope_geometry.py) | `RopeGeometry` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/route_guidance.py](src/prince_valiant_rl/route_guidance.py) | `distance_potential`, `build_corridor`, `stone_phase_key`, `build_phase_corridor`, `RouteGuidance` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/route_waypoints.py](src/prince_valiant_rl/route_waypoints.py) | `demonstrated_goals`, `RouteWaypoints` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/scene_state.py](src/prince_valiant_rl/scene_state.py) | `ForestTerrain` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/score_clear_capture.py](src/prince_valiant_rl/score_clear_capture.py) | `ScoreClearRing` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/baseline_gpu_control.py](src/prince_valiant_rl/training/baseline_gpu_control.py) | `CurrentPath`, `BaselineCudaControl`, `ArchivedCudaPolicy` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/checkpoint_io.py](src/prince_valiant_rl/training/checkpoint_io.py) | `CheckpointFinalizationError`, `_file_sha256`, `_path_diagnostics`, `_feed`, `checkpoint_state_sha256`, `_saved_state_matches`, `atomic_torch_save` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/dqfd.py](src/prince_valiant_rl/training/dqfd.py) | `paired_return`, `paired_targets`, `dqfd_objective`, `GlobalPER`, `OwnerNStep`, `LosslessFrameRing`, `build_q_from_actor`, `optimizer_for`, `optimizer_step`, `output_comparison`, `OnlineCadence`, `GreedyQPolicy`, `evaluate_greedy_episode` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_continuation.py](src/prince_valiant_rl/training/dqfd_continuation.py) | `digest`, `validate_budget`, `restore_payload`, `create_continuation_learner`, `run_continuation` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_evaluation.py](src/prince_valiant_rl/training/dqfd_evaluation.py) | `evaluate_reset389_video` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_execution.py](src/prince_valiant_rl/training/dqfd_execution.py) | `state_hash`, `_tensor`, `_stack_windows`, `_online_row`, `gather_mixed_batch`, `LearnerCounters`, `DQfDBatchLearner`, `FixedBCollector` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_level1_adapter.py](src/prince_valiant_rl/training/dqfd_level1_adapter.py) | `_read_header`, `_NpyRows`, `_ArchiveRows`, `_CountingNative`, `Level1DQfDAdapter`, `create_adapter` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_pretrain_entrypoint.py](src/prince_valiant_rl/training/dqfd_pretrain_entrypoint.py) | `production_code_hash`, `_counter_snapshot`, `_elapsed_ticks`, `check_original_total_deadline`, `_canonical_receipt_hash`, `validate_execution_budget_receipt`, `load_execution_budget_receipt`, `ParentResourceEvidenceError`, `verify_parent_work_evidence`, `verify_accepted_handoff`, `_publish_parent_acceptance`, `_publish_immutable_json`, `validate_launch`, `private_child`, `dispatch_pretrain` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_pretraining.py](src/prince_valiant_rl/training/dqfd_pretraining.py) | `learner_state_hashes`, `_save_milestone`, `load_committed_milestone`, `_rank`, `_retention_passed`, `run_demo_pretraining` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_process_ownership.py](src/prince_valiant_rl/training/dqfd_process_ownership.py) | `process_record`, `validate_private_child_ownership` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_production.py](src/prince_valiant_rl/training/dqfd_production.py) | `_window_digest`, `_pinned_original_389_pre_window`, `_same_array_exact`, `_same_window_exact`, `production_reset389_episode`, `_production_reset389_episode`, `_sha256`, `_validate_inputs`, `create_fresh_demo_learner` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_receipts.py](src/prince_valiant_rl/training/dqfd_receipts.py) | `_fsync_receipt`, `_canonical`, `serialize_record`, `append_record`, `_parse_records`, `read_records`, `latest_record`, `IncrementalRecordReader` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_replay.py](src/prince_valiant_rl/training/dqfd_replay.py) | `DemoRow`, `DemoReplayIndex` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_study_budget.py](src/prince_valiant_rl/training/dqfd_study_budget.py) | `canonical_hash`, `_read_bound`, `validate_extended_receipt`, `validate_loop_limits`, `caps_for_budget` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_supervision.py](src/prince_valiant_rl/training/dqfd_supervision.py) | `PERFORMANCE_INFORMATION`, `resource_sample`, `gpu_free_memory_sample`, `ml_resource_sample`, `validate_ml_resource_sample`, `validate_resource_sample`, `_append_bounded`, `_artifact_bytes`, `supervise_replay`, `_atomic_receipt` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqfd_transitions.py](src/prince_valiant_rl/training/dqfd_transitions.py) | `frame_index_capacity`, `progress_receipt_budget`, `TransitionError`, `sha256_file`, `_summary`, `compare_window`, `lossless_cast`, `boundary`, `FrameLedger`, `_atomic_json`, `_guard`, `_accounting_snapshot`, `_reserve_growth`, `_commit_growth`, `_reconcile_files`, `_append_outcome`, `_uncertain_frames`, `_completed_frames`, `_validate_outcomes`, `build_cache`, `_WindowStore`, `_FrameIndexedWindowStore`, `_validate_measurements`, `_FrameIndexedReader`, `TransitionCache`, `inspect_npz` | Duration-aware DQfD loss/replay/validation, supervised training or strict checkpoint/evaluation contract |
| [src/prince_valiant_rl/training/dqn.py](src/prince_valiant_rl/training/dqn.py) | `DuelingQ`, `masked_argmax`, `double_target`, `storage_dtype`, `WindowReplay`, `NStep`, `optimize_q`, `load_q` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/evaluation_capture.py](src/prince_valiant_rl/training/evaluation_capture.py) | `atomic_write_json`, `_canonical_arrays`, `policy_input_schema`, `fingerprint_policy_input`, `fingerprint_reference_record`, `fingerprints_equal`, `BoundedJsonlWriter`, `StorageGuardInterruption`, `ArtifactBudgetInterruption`, `ArtifactBudgetGuard`, `disk_measurements`, `DiskGuard`, `ReferenceFingerprintReader`, `assess_storage`, `partial_capture_status`, `iter_json_array`, `build_reference_fingerprint_bundle` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/events.py](src/prince_valiant_rl/training/events.py) | `word`, `section`, `live_position`, `EventTracker`, `reward_components`, `bootstrap_allowed`, `duration_gae` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/input_features.py](src/prince_valiant_rl/training/input_features.py) | `InputFeatureEncoder`, `configure_encoder`, `upgrade_model`, `augment_unavailable_demonstrations`, `attach_recorded_demonstrations` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/interface.py](src/prince_valiant_rl/training/interface.py) | `action_name`, `load_catalog`, `FeatureEncoder`, `TrainingEnv` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/level2_completion_evidence.py](src/prince_valiant_rl/training/level2_completion_evidence.py) | `region`, `CompletionEvidence`, `bonus_event` | Level2 data/actor/reward/completion evidence dependency |
| [src/prince_valiant_rl/training/level2_demonstration_policy.py](src/prince_valiant_rl/training/level2_demonstration_policy.py) | `DemonstrationActor`, `KnownStatePolicy` | Level2 data/actor/reward/completion evidence dependency |
| [src/prince_valiant_rl/training/level2_model_contract.py](src/prince_valiant_rl/training/level2_model_contract.py) | `load_contract`, `create_fresh_level2_model`, `parameter_signature` | Level2 data/actor/reward/completion evidence dependency |
| [src/prince_valiant_rl/training/level2_runtime.py](src/prince_valiant_rl/training/level2_runtime.py) | `checked_ram`, `transition`, `Encoder`, `History`, `ten_step_arrays` | Level2 data/actor/reward/completion evidence dependency |
| [src/prince_valiant_rl/training/level2_score_reward.py](src/prince_valiant_rl/training/level2_score_reward.py) | `ScoreDiscontinuity`, `decode_score`, `transition` | Level2 data/actor/reward/completion evidence dependency |
| [src/prince_valiant_rl/training/level3_evidence.py](src/prince_valiant_rl/training/level3_evidence.py) | `match_reference_region` | Raw-RAM imitation or independent Level3 evidence |
| [src/prince_valiant_rl/training/level3_imitation.py](src/prince_valiant_rl/training/level3_imitation.py) | `observe_ram`, `Level3ImitationActor` | Raw-RAM imitation or independent Level3 evidence |
| [src/prince_valiant_rl/training/models.py](src/prince_valiant_rl/training/models.py) | `tensor_observation`, `ActorCritic`, `parameter_hash`, `freeze` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/process_envs.py](src/prince_valiant_rl/training/process_envs.py) | `child_main`, `ProcessEnvs`, `grouped_advantages`, `condition_critic`, `resident_bytes` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [src/prince_valiant_rl/training/scene_pool.py](src/prince_valiant_rl/training/scene_pool.py) | `SequentialScenePool` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/training/scenes.py](src/prince_valiant_rl/training/scenes.py) | `scene_catalog`, `ControlledScene` | Native state, semantic/RAM observation or action/reset pipeline |
| [src/prince_valiant_rl/training/starts.py](src/prince_valiant_rl/training/starts.py) | `CachedStart` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/temporal.py](src/prince_valiant_rl/training/temporal.py) | `offsets_checked`, `empty_observation`, `FrameHistory`, `TemporalEnv` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [src/prince_valiant_rl/training/temporal_models.py](src/prince_valiant_rl/training/temporal_models.py) | `configure_precision`, `cuda_device`, `window_tensors`, `indexed_window_tensors`, `infer_shapes`, `masked_distribution`, `WindowPath`, `TemporalActorCritic`, `save_temporal`, `load_temporal` | Level1 observation/history, models, PPO curricula or worker lifecycle |
| [src/prince_valiant_rl/training/visual.py](src/prince_valiant_rl/training/visual.py) | `preprocess`, `image_stack`, `VisualLevel1Env`, `VisualEncoder`, `VisualPredictor`, `VisualAuxiliary`, `balanced_bce`, `prediction_loss`, `labels`, `VisualWindowPath`, `VisualActorCritic`, `optimizer_groups`, `parameter_digest` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
| [src/prince_valiant_rl/training/window_storage.py](src/prince_valiant_rl/training/window_storage.py) | `compact_window`, `materialize_observation_batch` | Transitive runtime dependency; direct importer(s) recorded in manifests/dependency_graph.json |
