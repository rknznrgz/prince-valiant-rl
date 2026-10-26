# Prince Valiant submission demo videos

Start with **progress_presentation.mp4**. This is a 3–5 minute, 1280×960,
30 FPS H.264 MP4 assembled from recorded agent gameplay. The complete game
image retains its source aspect ratio; labels sit outside the gameplay.
No voiceover or game audio was available in these source recordings.

## Contents

- `progress_presentation.mp4`: early and middle Level 1 DQN/DQfD checkpoints
  at 1,000 and 2,000 optimizer updates, then complete selected final runs of
  Levels 1–3. Early/middle runs play at 24× and final runs at 8×, with speeds
  displayed. Each source run is continuous from beginning to end, with no
  internal edits; acceleration samples frames uniformly. Cuts occur only
  between runs and presentation cards.
- `level1_full_run_1x.mp4`: selected DQN/DQfD update 3,000, complete real-time
  run, dynamic running reward, natural level completion.
- `level2_full_run_1x.mp4`: final title-inclusive neural imitation policy,
  complete real-time run through visible Level 2 completion and Level 3 entry.
- `level3_full_run_1x.mp4`: final neural imitation policy, complete real-time
  run through visible Level 3 completion and the Level 4 title.
- `manifest.json`: exact source paths, source/output SHA-256 hashes, playback
  speeds, checkpoint/reward labels, resolution, frame rate, durations and
  full-decode verification results.
- `build_videos.py` and `build/`: reproducible packaging script, intermediate
  segments, title cards, overlay text and FFmpeg logs.

## What the footage establishes

The Level 1 comparison uses the same demonstration-assisted DQN learning
lineage and unchanged reward weights. Update 1,000 timed out with reward
-22.963344, update 2,000 timed out with -14.202617, and selected update 3,000
cleared with +108.091791. Update 4,000 also cleared; update 5,000 regressed.
"Final" means the selected successful checkpoint, not the latest update.
Training counts are optimizer updates, not environment frames. The Level 2
8,000 counter describes its title-inclusive fitting branch, not cumulative
project training. Level 3 includes 20,000 initial plus 15,500 refinement updates.

Level 1 reward overlays update with the running sum of native-transition
rewards at the displayed gameplay frame. The frozen DQN checkpoints were replayed
without training to recover these histories; action-trace, capture-timestamp,
video SHA-256, terminal, episode length and final reward are checked against
the original receipts. Level 2 running reward is reconstructed from its saved
consecutive RAM with the documented `level2_score_delta_v4` function: completion,
game-over, energy loss, elapsed time and weighted score delta. The incoming title
is excluded until fresh gameplay, and reward is held after Level 2 completion
so the next-level tail does not earn Level 2 rewards. This is post-hoc evaluation
arithmetic, not the imitation model's training objective. Level 3 has no defined
task reward and remains marked N/A. The recurring bottom evidence disclaimer
has been removed from all videos.

Separate policies and independently initialized level runs are used. This is
not one uninterrupted game from Level 1 through Level 3. No training or gameplay
intervention was performed. Three frozen Level 1 replays recover reward histories;
Level 2 calculations use saved RAM without emulator execution.
These are known-route demonstrations; no generalization or 7/10
acceptance claim is made. Level 1's saved reset-389 completion is real, but
normal-game-start compliance remains unaudited in its source study.

The brief requests one 3–5-minute presentation with full level runs. These
existing runs total about 26 minutes in real time, so the short presentation
uses explicitly marked acceleration and supplies real-time companions. The
brief does not explicitly settle whether accelerated playback is acceptable.
Level 2 reward reconstruction, absent Level 3 reward and formal evaluation
requirements are disclosed limitations; this package does not claim full task acceptance.

## Evidence

All paths below are relative to the sibling `prince-valiant-rl` directory:

- `analysis/level1_dqfd_extended_study_20261025/README.md`
- `analysis/level1_dqfd_demo_continuation_20261025/README.md`
- `analysis/level1_dqfd_demo_continuation_20261025/selected_checkpoint.json`
- `analysis/level2_standalone_completion_v58_20261026/README.md`
- `analysis/level2_transition_tail_v59_20261026/README.md`
- `analysis/level3_imitation_evaluation_v2_20261026/README.md`

Rebuild from `RL Dev Task` using:

```powershell
.\prince-valiant-rl\.venv-gpu\Scripts\python.exe -X utf8 submission_demo_videos/prepare_running_rewards.py
.\prince-valiant-rl\.venv-gpu\Scripts\python.exe -X utf8 submission_demo_videos/build_videos.py
```

This overwrites only generated files in this video package and reads the
original project evidence without modifying it.
