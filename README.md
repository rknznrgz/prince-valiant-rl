# Prince Valiant — RAM-based reinforcement learning

A Windows/Python research project for **The Legend of Prince Valiant (NES)**. It combines a native emulator bridge, RAM observations, demonstration-assisted reinforcement learning, and separate imitation policies for later levels.

## Results

| Level | Selected method | Recorded result |
| --- | --- | --- |
| 1 | Dueling DQN trained with DQfD | Update 3,000 completed the level in 18,273 native frames; return +108.091791 |
| 2 | Neural behavior cloning | Completion card and subsequent Level 3 entry verified |
| 3 | Raw-RAM neural imitation | Completion card and subsequent Level 4 title verified |

These are known-route results with separate models and reset states. They do not establish the assignment's 7/10 success criterion. Level 1 evaluation from the game's normal start remains unaudited. See [results and evidence](docs/RESULTS.md).

## Getting started

Windows x64, Python 3.13, and NVIDIA CUDA are required by the selected training and gameplay runners. Checkpoints can be inspected on CPU.

```powershell
py -3.13 -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-cuda.txt
python run.py --list
python run.py inspect_models
```

The repository includes selected checkpoints, logs, code, and videos. ROMs, recordings, training datasets, reset states, and compiled DLLs are not included. Evaluation and training require these additional inputs; see [setup and commands](docs/SETUP.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md): native bridge, RAM processing, models, and learning code.
- [Results](docs/RESULTS.md): selected checkpoints, measured outcomes, and evidence.
- [Reproducibility](docs/REPRODUCIBILITY.md): verification, required assets, and current limitations.
- [Case study](docs/case_study/report.pdf): revised report, 10 main pages plus appendices; [editable source and build instructions](docs/case_study/README.md).
- [Progress video](deliverables/videos/progress_presentation.mp4): 4m23.67s presentation; [full-run videos](deliverables/videos/README.md).

Third-party notices and unresolved licensing questions are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project-wide license has been selected.
