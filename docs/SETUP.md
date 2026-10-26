# Setup and commands

Run commands from the repository root. Use a new output directory for every run.

## Python environment

The supported development environment is Windows x64 with CPython 3.13. Selected training and evaluation scripts use CUDA; model inspection uses CPU. Dependency pins reflect the observed environment. A fresh-machine installation has not been verified.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
python run.py --list
python run.py inspect_models
python run.py submission_assets audit
```

## Native emulator and ROM

Install Visual Studio Build Tools with MSVC x64 C++14 support, the Windows SDK, and Visual Studio Installer's `vswhere.exe`. Build the recorder before its ABI2 observer:

```powershell
python instrumentation/build_recorder.py
python instrumentation/build_kind18_observer.py
```

The retained native source has an unresolved licensing provenance issue; see [third-party notices](../THIRD_PARTY_NOTICES.md). No compiled binary is included.

Place a legally obtained ROM at `roms/prince_valiant.nes`. Expected SHA-256:

```text
96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5
```

Reset states are bound to the ROM and recorder build identity. Level 1 also checks exact historical DLL and observer-provenance hashes. A fresh build is not verified as compatible with those assets: the original build fingerprint included native tests and benchmarks omitted here. Historical evaluation requires the original compatible assets or a separately validated codec compatibility procedure.

## Required assets

[artifacts.json](../manifests/artifacts.json) records required filenames, sizes, hashes, destinations, and access conditions. Public downloads are not available. The original workspace owner can restore individual local assets:

```powershell
python run.py submission_assets restore --source <research-checkout> --destination <exact-manifest-destination>
```

Restoration rejects existing destinations and hash mismatches. Generating new recordings does not reproduce the selected experiments' pinned identities. Preparation commands can write to fresh locations, but historical consumers still expect their documented destinations and hashes.

## Evaluation

After supplying the required data, states, ROM, and compatible native binaries:

```powershell
python run.py evaluate_selected_level1 --output analysis/submission_l1_eval
python run.py evaluate_level2_confirmed_completion --output analysis/submission_l2_eval
python run.py evaluate_level3_imitation --checkpoint training/level3_imitation_refine_v2_20261026/best.pt --output analysis/submission_l3_eval
python run.py review_level3_imitation --output analysis/submission_l3_eval
```

Level 1 uses the recorded reset-389 protocol. Level 2 checks the completion card and subsequent title. Level 3 independently recomputes every model action and compares ending images against retained references. These commands have been help-checked; no new gameplay was run to verify this repository export.

## Training

### Level 1 DQfD

```powershell
python run.py level1_dqfd_pilot preflight --output training/submission_dqfd
python run.py level1_dqfd_pilot pretrain --output training/submission_dqfd --execution-budget config/submission_execution_budget.json --report-dir analysis/submission_dqfd_receipts --dry-run
```

The execution-budget file is a required future input, not an included artifact. Training requires the pinned demonstration cache, initial actor, and a fresh valid budget/code/protocol chain. Remove `--dry-run` only after those requirements are met. Historical receipts cannot be reused to authorize a new launch. The supported continuation restores Q/target networks, Adam, RNG, and replay priorities from the accepted update-2000 state; arbitrary checkpoint resume and online collection are not exposed by this CLI.

### Level 1 PPO

```powershell
python run.py train_backward_level1 --stage train --config config/backward_level1_v1.json --output training/submission_ppo --updates 1
python run.py train_backward_level1 --stage evaluate --config config/backward_level1_v1.json --output analysis/submission_ppo_eval --resume training/submission_ppo/checkpoints/latest.pt --trials 10 --evaluation-mode greedy
```

This base-config example exercises the retained backward PPO workflow. It is not a reproduction of every historical PPO variant or the selected DQfD result. Route and config-specific normalization/BC inputs are required.

### Levels 2 and 3 imitation

```powershell
python run.py run_level2_title_bc --output training/submission_l2_title --report analysis/submission_l2_title
python run.py train_level3_imitation --output training/submission_l3_initial --steps 20000
python run.py train_level3_imitation --output training/submission_l3_refine --resume training/submission_l3_initial/best.pt --steps 15500 --lr 0.00001
```

Level 2 requires the v56 parent, v4 RAM replay, title preparation, and original recording. Level 3 requires its pinned preparation and demonstration dataset. Its `--resume` option loads actor state and starts a fresh Adam optimizer; it is refinement rather than exact optimizer continuation.

## Data preparation

```powershell
python run.py prepare_backward_level1 --session captures/human/20261013_224343_044974 --output training/submission_route
python run.py prepare_level2_replay --output analysis/submission_l2_replay
python run.py prepare_level2_score_replay --output analysis/submission_l2_score_replay
python run.py prepare_level2_title_bc --output analysis/submission_l2_intro
python run.py prepare_level3_imitation --output analysis/submission_l3_preparation
python run.py prepare_level1_dqfd_transitions --mode preflight --output training/submission_transitions --frame-cap 100000 --artifact-cap-bytes 12884901888 --wall-cap-seconds 1800
```

The DQfD builder's replay mode additionally requires an adapter, fresh resource receipt, and separate lifecycle receipt directory; see `python run.py prepare_level1_dqfd_transitions --help`. Replay and some state-preparation tools execute the emulator. The 100,000-frame preflight example is a bounded parser example, not a validated budget for the complete reconstruction.
