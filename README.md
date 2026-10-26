# Prince Valiant: curated local submission preparation

Prepared 26 October 2026. This curated research submission is public at [https://github.com/rknznrgz/prince-valiant-rl](https://github.com/rknznrgz/prince-valiant-rl). It is not a fully
compliant assignment submission; the documented evidence and reproducibility gaps remain. See [github.md](github.md)
for implementation review, dependency maps, verification and blockers.

Use Windows x64 and CPython 3.13. Selected training and gameplay runners require
NVIDIA CUDA (observed torch 2.14.1+cu130); CPU is supported for checkpoint inspection.
Other platforms are unverified. The installed package versions were read locally;
their availability on a fresh package index has not been verified.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
```

The native source route requires Visual Studio Build Tools with MSVC x64 C++14,
Windows SDK and the standard Visual Studio Installer vswhere.exe. Review
[third-party notices](THIRD_PARTY_NOTICES.md) before redistributing native code.
Build the recorder first, then its ABI2 observer:

```powershell
python instrumentation/build_recorder.py
python instrumentation/build_kind18_observer.py
```

No generated DLL is shipped. The state codec build ID must match saved states.
The Level 1 production evaluator also requires the original DLL/provenance SHA256
identities: a new compiler build may fail that check. Do not remove identity guards;
restore exact locally held binaries (rights permitting) or perform a separately
reviewed compatibility study. Builds were not run during packaging.

Supply a legally obtained ROM at `roms/prince_valiant.nes`. SHA256 must be
`96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5`.
The recorder verifies this identity; `submission_assets audit` checks it without
loading native code. No cartridge bytes are included.

Required local-only inputs are listed with size, hash and exact destination in
[artifact manifest](manifests/artifacts.json). There are no public download URLs.
For each reviewed input, the original owner can restore it explicitly:

```powershell
python run.py submission_assets restore --source <research-checkout> --destination <exact-manifest-destination>
```

This is an optional local preparation step, not an instruction to publish
recordings or reset states. It rejects occupied destinations and wrong hashes.
New recordings can be captured using another recorder, but do not satisfy the
selected studies' pinned data identities; exact training reproduction currently
requires owner-provided inputs. The preparation tools document extraction from
the original selected ancestry. Do not treat all copied checkpoints as proof
that the package is training-ready.

| Workflow | Included functionality | Current readiness |
|---|---|---|
| Level 1 DQN/DQfD | Exact update-3000 checkpoint, committed receipt, DuelingQ, losses/PER, transition builder, supervised trainer and exact continuation loader, reset-389 evaluator | CPU checkpoint load verified; native evaluation blocked on route/states, identity-pinned DLL/provenance and ROM; training additionally needs pinned replay/actor/budget chain |
| Level 1 PPO | Backward PPO trainer/evaluator, process workers, curricula, rewards, temporal models | Source/help checked; route, normalization/BC/config-specific inputs required; no selected PPO result replaced by this package |
| Level 2 imitation | Title-inclusive fitting from v56 parent; neural completion runner and independent visual card/title check | Model load verified; exact data/parent/reset and native inputs withheld; not training/evaluation ready as distributed |
| Level 3 imitation | Raw-RAM fitting/refinement, selected actor, current-RAM evaluator and independent all-action/reference review | Model load verified; normalization embedded; exact preparation/dataset/reset/reference images needed for evaluation/review |
| General environment | Gymnasium RAM wrapper, ROM identity, extraction, actions and state loading | Source/help checked; nes-py wheel and gameplay state required; legacy wrapper differs from selected RecorderNative workflows |

```powershell
python run.py --list
python run.py inspect_models
python run.py submission_assets audit
# After supplying assets and building native code; these commands were NOT executed:
python run.py evaluate_selected_level1 --output analysis/submission_l1_eval
python run.py evaluate_level2_confirmed_completion --output analysis/submission_l2_eval
python run.py evaluate_level3_imitation --checkpoint training/level3_imitation_refine_v2_20261026/best.pt --output analysis/submission_l3_eval
python run.py review_level3_imitation --output analysis/submission_l3_eval
python run.py train_level3_imitation --output training/submission_l3_initial --steps 20000
python run.py train_level3_imitation --output training/submission_l3_refine --resume training/submission_l3_initial/best.pt --steps 15500 --lr 0.00001
python run.py run_level2_title_bc --output training/submission_l2_title --report analysis/submission_l2_title
python run.py prepare_level2_title_bc --output analysis/submission_l2_intro
python run.py prepare_level2_replay --output analysis/submission_l2_replay
python run.py prepare_level2_score_replay --output analysis/submission_l2_score_replay
python run.py prepare_level3_imitation --output analysis/submission_l3_preparation
python run.py prepare_backward_level1 --session captures/human/20261013_224343_044974 --output training/submission_route
python run.py train_backward_level1 --stage train --config config/backward_level1_v1.json --output training/submission_ppo --updates 1
python run.py train_backward_level1 --stage evaluate --config config/backward_level1_v1.json --output analysis/submission_ppo_eval --resume training/submission_ppo/checkpoints/latest.pt --trials 10 --evaluation-mode greedy
python run.py level1_dqfd_pilot preflight --output training/submission_dqfd
python run.py level1_dqfd_pilot pretrain --output training/submission_dqfd --execution-budget config/submission_execution_budget.json --report-dir analysis/submission_dqfd_receipts --dry-run
python run.py prepare_level1_dqfd_transitions --mode preflight --output training/submission_transitions --frame-cap 100000 --artifact-cap-bytes 12884901888 --wall-cap-seconds 1800
python run.py prepare_level1_dqfd_transitions --mode replay --output training/submission_transitions --frame-cap 100000 --artifact-cap-bytes 12884901888 --wall-cap-seconds 1800 --adapter prince_valiant_rl.training.dqfd_level1_adapter:create_adapter --resource-receipt analysis/submission_resource_gate.json --receipt-dir analysis/submission_replay_receipts
```

Commands above are actual parser examples, not results from new gameplay.
`config/submission_execution_budget.json` and `analysis/submission_resource_gate.json`
are required future reviewed inputs, NOT supplied or fabricated artifacts. The
DQfD production training command is the shown pretrain command with `--dry-run`
removed, once its input/authority/resource blockers are resolved. Transition
replay advances the emulator; it was not executed during packaging. The100000
frame cap is a bounded example and may need a reviewed budget appropriate to
all selected replay sources. Exact acceptance is not inferred from parser help.
Always choose fresh output directories. Preparation outputs use fresh names but
historical consumers still require their pinned artifact destinations; validate
the generated identity before installing it there. The PPO `latest.pt` example
must be replaced with the actual checkpoint filename written by the chosen run.
DQfD `pretrain` requires a fresh valid `--execution-budget` and `--report-dir`;
see `python run.py level1_dqfd_pilot pretrain --help`. Historical receipts are not
new authorization. Generic unchecked resume/online/final-eval phases are not
offered. Level 3 `--resume` loads actor state, then starts a fresh Adam optimizer;
it is refinement, not exact optimizer continuation. Level 2 requires its v56 parent.

The 10-page [report PDF](docs/case_study_draft/REPORT_DRAFT.pdf) is explicitly a
draft. No revised LaTeX PDF was found in the documented report location.
[Presentation](deliverables/videos/progress_presentation.mp4) is 4m23.67s,
1280x960, 30fps H.264, with three real-time full-run companions. Early/middle
runs are 24x and selected runs 8x; separate resets/models are used. No voiceover
or game audio is present. Level 2/3 episode rewards are unavailable, shown N/A.
The brief does not settle accelerated playback acceptance.

Tests, private recordings, ROMs, caches, virtual environments, obsolete studies
and agent tooling are excluded. Originals remain in the research workspace.
Models are each below GitHub's 100 MiB file limit; two companion videos exceed
50 MiB and need a deliberate storage decision before publication. No LFS pointers
or fabricated release links are included.

Rebuild into an unused directory from the original workspace with the original
Python environment (needed only to record installed dependency metadata):

```powershell
python packaging/prepare_export.py --source <research-checkout> --output <new-empty-path>
```

This uses an explicit entry-point dependency closure and artifact allowlist,
never broad source/artifact tree copies. No training or evaluation is run.
