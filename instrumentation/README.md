# Native emulator integration

`recorder_api.cpp` implements the recorder bridge and build-bound state serialization. The kind-18 observer adds independently versioned combat-event instrumentation. `nes-py/nes_emu/` contains the retained native runtime sources.

Build order: `python instrumentation/build_recorder.py`, then `python instrumentation/build_kind18_observer.py`. Windows x64 MSVC and SDK prerequisites, historical reset compatibility, and required assets are documented in [setup](../docs/SETUP.md).

Compiled DLLs and generated build trees are excluded. Read [third-party notices](../THIRD_PARTY_NOTICES.md) and the retained nes-py licensing files before redistribution.
