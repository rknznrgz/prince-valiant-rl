# Learning components

Reusable temporal models, replay stores, losses, process workers, checkpoint validation, and level-specific imitation actors.

- `dqfd*.py` and `dqn.py`: demonstration-assisted Double-Q learning, prioritized replay, supervision, and strict state restoration.
- `temporal*.py`, `process_envs.py`, and `window_storage.py`: history encoding and batched training support.
- `level2_demonstration_policy.py` and `level3_imitation.py`: the selected raw-RAM imitation actors.

These are implementation modules. Trained weights and logs are in the repository-root [training directory](../../../training/). See [architecture](../../../docs/ARCHITECTURE.md) for model contracts.
