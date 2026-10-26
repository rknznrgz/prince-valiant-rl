"""Lazy Level1Env adapter and single-pass ZIP/NPY source reader for DQfD replay.

Importing this module is data-only. ``create_adapter`` is the only function
that imports and constructs Level1Env (and therefore loads the native backend).
It is reachable only through the CLI's explicit ``--mode replay`` path.
"""
from __future__ import annotations

import json
import math
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .dqfd_transitions import (ROW_COUNT, SOURCES, TransitionError, frame_index_capacity,
                               sha256_file)
from .dqfd_receipts import latest_record

ROOT = Path(__file__).resolve().parents[3]
METADATA_FIELDS = {"frame", "action", "action_mask", "confidence", "changed_action",
                   "previous_route_mask", "post_ram_sha256", "provenance_kind", "trajectory_id"}


def _read_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version in ((2, 0), (3, 0)):
        return np.lib.format.read_array_header_2_0(stream)
    raise TransitionError(f"unsupported source NPY version: {version}")


class _NpyRows:
    def __init__(self, archive: zipfile.ZipFile, key: str, expected_rows: int):
        self.stream = archive.open(key + ".npy", "r")
        try:
            shape, fortran, dtype = _read_header(self.stream)
            self.shape, self.dtype = tuple(shape), np.dtype(dtype)
            if not self.shape or self.shape[0] != expected_rows or fortran or self.dtype.hasobject:
                raise TransitionError(f"unsupported or unexpected NPY layout for {key}")
            self.row_shape = self.shape[1:]
            self.row_bytes = int(np.prod(self.row_shape, dtype=np.int64)) * self.dtype.itemsize
        except BaseException:
            self.stream.close()
            raise

    def read(self):
        raw = self.stream.read(self.row_bytes)
        if len(raw) != self.row_bytes:
            raise TransitionError("truncated source NPY member")
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.row_shape).copy()

    def close(self):
        self.stream.close()


class _ArchiveRows:
    """Keep one decompressor per member and advance every member once in order."""
    def __init__(self, path: Path, *, expected_rows: int = ROW_COUNT):
        self.expected_rows = int(expected_rows)
        self.archive = zipfile.ZipFile(path, "r")
        self.arrays = {}
        try:
            members = {Path(name).stem for name in self.archive.namelist() if name.endswith(".npy")}
            self.observation_fields = tuple(sorted(members - METADATA_FIELDS))
            required = set(self.observation_fields) | {"frame", "action", "action_mask", "post_ram_sha256",
                                                       "provenance_kind", "trajectory_id"}
            if not self.observation_fields or not required.issubset(members):
                raise TransitionError("source archive members do not satisfy the active observation/label contract")
            for key in sorted(required):
                self.arrays[key] = _NpyRows(self.archive, key, self.expected_rows)
        except BaseException:
            for array in self.arrays.values():
                array.close()
            self.archive.close()
            raise
        self.cursor = 0
        self.window_bytes = sum(self.arrays[key].row_bytes for key in self.observation_fields)
        validity = self.arrays.get("history_valid")
        if validity is None and self.expected_rows != ROW_COUNT:
            # Small row-reader fixtures exercise ZIP streaming, not the live schema.
            pass
        elif validity is None or validity.row_shape != (7,):
            raise TransitionError("archive history_valid header is not exactly seven slots")
        self.temporal_fields = set()
        self.non_temporal_fields = set()
        self.headers = {}
        for key in self.observation_fields:
            member = self.arrays[key]
            self.headers[key] = {"shape": list(member.shape), "dtype": member.dtype.str,
                                 "nbytes": int(member.row_bytes * self.expected_rows)}
        if validity is not None:
            self.temporal_fields.add("history_valid")
        for key in set(self.observation_fields) - {"history_valid"}:
            shape = self.arrays[key].row_shape
            if shape and shape[0] == 7:
                self.temporal_fields.add(key)
            else:
                self.non_temporal_fields.add(key)

    def read(self):
        if self.cursor >= self.expected_rows:
            raise TransitionError("source archive contains extra rows")
        raw = {key: member.read() for key, member in self.arrays.items()}
        self.cursor += 1
        pre = {key: raw[key] for key in self.observation_fields}
        return {"pre": pre, "frame": int(raw["frame"]), "action": int(raw["action"]),
                "action_mask": int(raw["action_mask"]), "source_kind": str(raw["provenance_kind"]),
                "trajectory_id": int(raw["trajectory_id"]), "source_id": int(raw["trajectory_id"]),
                "post_ram_sha256": str(raw["post_ram_sha256"])}

    def source_rows(self, trajectory_id: int, count: int):
        expected_id = trajectory_id
        pending = None
        for local in range(count):
            current = pending if pending is not None else self.read()
            pending = None
            if current["trajectory_id"] != expected_id:
                raise TransitionError(f"source trajectory ID mismatch at global row {self.cursor - 1}")
            if local + 1 < count:
                pending = self.read()
                if pending["trajectory_id"] != expected_id:
                    raise TransitionError(f"source ID changed inside trajectory {expected_id}")
                current["next_pre"] = pending["pre"]
            else:
                current["next_pre"] = None
            yield current

    def close(self):
        for array in self.arrays.values():
            array.close()
        self.archive.close()


class _CountingNative:
    """Per-instance proxy that records native calls even when env postprocessing fails."""
    def __init__(self, owner):
        self._owner = owner
        self.phase = "reset"
        self.counts = {key: {"planned": 0, "attempted": 0, "completed": 0, "uncertain": 0}
                       for key in ("reset", "context", "labeled")}
        for row in self.counts.values():
            row.update(planned_actions=0, attempted_actions=0, completed_actions=0, uncertain_actions=0)

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def frame_advance(self, mask):
        row = self.counts[self.phase]
        row["planned"] += 1
        row["attempted"] += 1
        try:
            result = self._owner.frame_advance(mask)
        except BaseException:
            row["uncertain"] += 1
            raise
        row["completed"] += 1
        row["planned"] -= 1
        return result


class Level1DQfDAdapter:
    def __init__(self, env: Any, archive_path: Path, *, rows: _ArchiveRows,
                 frame_cap: int, artifact_cap: int, wall_cap: float,
                 history_offsets=(-60, -30, -15, -9, -6, -3, 0),
                 resource_sample_path: Path | None = None, output_root: Path | None = None):
        self.env, self.archive_path = env, Path(archive_path)
        self._native_owner = env.native
        self._counting_native = _CountingNative(self._native_owner)
        env.native = self._counting_native
        self._observer_pin = None
        self.frame_cap, self.artifact_cap, self.wall_cap = frame_cap, artifact_cap, wall_cap
        self.history_offsets = tuple(int(x) for x in history_offsets)
        self.resource_sample_path = Path(resource_sample_path) if resource_sample_path else None
        self.output_root = Path(output_root) if output_root else None
        self.rows: _ArchiveRows | None = rows
        self.temporal_fields = frozenset(getattr(rows, "temporal_fields", ()))
        self.non_temporal_fields = frozenset(getattr(rows, "non_temporal_fields", ()))
        # Full-size indexed pre-state memmaps, five final windows, plus a fixed
        # receipt/index/header allowance. Checked before constructing Level1Env.
        self.estimated_total_artifact_bytes = int(getattr(rows, "projected_cache_bytes", 0))
        self.estimated_next_write_bytes = 512 * 1024
        self.native_frames = 0
        self.frame_ledger = self._counting_native.counts
        self._source_by_name = {name: (i, count, reset, prefix, kind)
                                for i, (name, count, reset, prefix, kind) in enumerate(SOURCES)}

    def lineage(self):
        native = self.env.native
        observer = getattr(native, "abi_identity", None)
        if not isinstance(observer, dict):
            raise TransitionError("selected config requires observer ABI provenance, but loaded owner has none")
        observer_path = Path(observer["observer_dll_path"]).resolve()
        base_path = ROOT / "instrumentation/recorder_build/recorder_native.dll"
        provenance_path = Path(observer["provenance_path"]).resolve()
        if sha256_file(observer_path) != observer.get("observer_dll_sha256"):
            raise TransitionError("loaded observer DLL differs from its pinned ABI provenance")
        if self._observer_pin is not None:
            if (sha256_file(provenance_path) != self._observer_pin["provenance_sha256"]
                    or sha256_file(observer_path) != self._observer_pin["observer_dll_sha256"]
                    or observer.get("base_build_id") != str(native.build)):
                raise TransitionError("runtime observer ABI/provenance differs from preconstruction pins")
        if sha256_file(base_path) != observer.get("base_dll_sha256"):
            raise TransitionError("base codec DLL differs from loaded observer provenance")
        return {"rom_sha256": str(native.rom_hash), "native_build_id": str(native.build),
                "native_library_path": observer_path.relative_to(ROOT).as_posix(),
                "native_library_sha256": sha256_file(observer_path),
                "native_library_role": "loaded_kind18_observer",
                "observer_abi_identity": dict(observer),
                "observer_provenance_sha256": sha256_file(provenance_path),
                "base_native_library_path": base_path.relative_to(ROOT).as_posix(),
                "base_native_library_sha256": sha256_file(base_path),
                "route_sha256": str(self.env.route.meta["route_sha256"]),
                "route_metadata": "training/backward_level1_v1/route.json",
                "replay_contract": "Level1Env.reset(frame=verified_reset), authentic history/controller/reward state, step_mask(raw_mask)"}

    def trajectories(self):
        result = []
        for trajectory_id, (name, count, reset, prefix, kind) in enumerate(SOURCES):
            result.append({"name": name, "source_id": trajectory_id, "expected_rows": count, "reset_frame": reset,
                           "context_frames": prefix, "source_kind": kind,
                           "start_frame": reset + prefix, "expected_end_reason": "level_completed",
                           "rows": self.rows.source_rows(trajectory_id, count)})
        return result

    def planned_reset_frames(self, source):
        frame = int(source["reset_frame"])
        eligible = [int(value) for value in self.env.route.state_frames if int(value) <= frame]
        if frame not in eligible:
            raise TransitionError(f"exact pinned reset snapshot is unavailable for {source['name']}; fallback is forbidden")
        return 0

    def reset(self, source):
        self._counting_native.phase = "reset"
        reset_ledger = self.frame_ledger["reset"]
        reset_ledger["planned_actions"] += 1
        reset_ledger["attempted_actions"] += 1
        try:
            self.env.reset(frame=int(source["reset_frame"]), variation=0,
                           training_segment=False, episodic_life=False)
        except BaseException:
            reset_ledger["uncertain_actions"] += 1
            self.native_frames += self.frame_ledger["reset"]["completed"]
            raise
        reset_ledger["planned_actions"] -= 1
        reset_ledger["completed_actions"] += 1
        frames = int(self.env.replay_frames)
        self.native_frames = sum(x["completed"] for x in self.frame_ledger.values())
        if frames:
            raise TransitionError(f"{source['name']}: pinned exact reset unexpectedly replayed {frames} native frames")
        return {"native_frames": frames}

    def observation(self):
        return self.env.history.window()

    def step(self, raw_mask: int):
        category = "context" if getattr(self, "_context_dispatch", False) else "labeled"
        self._counting_native.phase = category
        action_ledger = self.frame_ledger[category]
        action_ledger["planned_actions"] += 1
        action_ledger["attempted_actions"] += 1
        before = self.frame_ledger[category]["completed"]
        try:
            _obs, reward, terminated, truncated, info = self.env.step_mask(int(raw_mask))
        except BaseException:
            action_ledger["planned_actions"] -= 1
            action_ledger["uncertain_actions"] += 1
            self.native_frames = sum(x["completed"] for x in self.frame_ledger.values())
            raise
        action_ledger["planned_actions"] -= 1
        action_ledger["completed_actions"] += 1
        duration = int(info["transition_frames"])
        actual = self.frame_ledger[category]["completed"] - before
        if actual != duration:
            raise TransitionError(f"owner-boundary frame count {actual} differs from environment duration {duration}")
        self.native_frames = sum(x["completed"] for x in self.frame_ledger.values())
        reason = str(info.get("end_reason") or "")
        natural = reason in ("game_over", "level_completed")
        time_limit = reason == "time_limit"
        if bool(terminated) != natural or (bool(truncated) and not time_limit and reason != "unsupported_mode"):
            raise TransitionError(f"Level1Env boundary flags disagree with end_reason={reason!r}")
        available = info.get("availability")
        reward_components = info.get("reward_components")
        if (not isinstance(available, dict) or not isinstance(reward_components, dict)
                or not set(reward_components).issubset(available)):
            raise TransitionError("production reward components and availability are missing or malformed")
        components = {str(key): float(value) for key, value in reward_components.items()}
        component_availability = {str(key): bool(available[key]) for key in reward_components}
        score = info.get("score_gain_coverage")
        if not isinstance(score, dict):
            raise TransitionError("production score coverage metadata is missing")
        score_state = dict(score)
        score_state["newly_triggered_quarantine"] = bool(score.get("quarantine_started_this_decision", False))
        score_state["measured_zero_native_frames"] = int(score.get("last_decision_measured_zero_native_frames", 0))
        score_state["applied_zero_native_frames"] = int(score.get("last_decision_zero_reward_native_frames", 0))
        reward_state = {
            "reset_frame": int(self.env.initial_frame), "elapsed_frames": int(self.env.elapsed),
            "game_elapsed_frames": int(self.env.game_elapsed), "life_index": int(self.env.life_index),
            "tracker_frontiers": [int(x) for x in self.env.tracker.frontiers],
            "tracker_visited": sorted(int(x) for x in self.env.tracker.visited),
            "tracker_objectives": sorted(int(x) for x in self.env.tracker.objectives),
            "tracker_completed": bool(self.env.tracker.completed),
            "reward_totals": {str(k): float(v) for k, v in info.get("reward_totals", {}).items()},
            "discounted_reward_totals": {str(k): float(v) for k, v in info.get("discounted_reward_totals", {}).items()},
            "discounted_return_valid": bool(info.get("discounted_return_valid", False)),
        }
        ram = np.asarray(self.env.ram, dtype=np.uint8).tobytes()
        if not math.isclose(float(reward), sum(components.values()), rel_tol=1e-7, abs_tol=1e-6):
            raise TransitionError("production reward differs from the sum of per-transition reward components")
        return {"observation": _obs, "info": {
                    "natural_terminal": natural, "time_limit": time_limit,
                    "game_terminal": bool(info.get("game_terminal", False)),
                    "end_reason": reason, "bootstrap_valid": bool(info["bootstrap_valid"]),
                    "transition_frames": duration},
                "duration": duration, "reward": float(reward), "components": components,
                "availability": component_availability, "score_state": score_state,
                "reward_state": reward_state,
                "ram": ram, "executed_mask": int(self.env.last_action),
                "terminal_observation_verified": True,
                "invalid_bcd": getattr(self.env, "_score_gain_failure", None) is not None}

    def check_runtime_guards(self):
        if self.native_frames > self.frame_cap:
            raise TransitionError("adapter native-frame count exceeded launch cap")
        if self.resource_sample_path is None:
            raise TransitionError("live replay has no parent-supervisor resource sample channel")
        stop_path = self.resource_sample_path.parent / "STOP_REQUESTED"
        if stop_path.exists():
            raise TransitionError("parent supervisor requested replay stop")
        try:
            launch = json.loads((self.resource_sample_path.parent / "reconstruction_resource_launch_receipt.json").read_text(encoding="utf-8"))
            sample = latest_record(self.resource_sample_path, attempt_id=str(launch["attempt_id"]))["payload"]
            age = time.time() - float(sample["sample_unix_time"])
            if age < 0 or age > 4.0 or sample.get("runtime_gate_status") != "passed":
                raise ValueError("supervisor sample is stale or failed")
            for key, floor in (("available_ram_bytes", 4 * 1024**3),
                               ("commit_headroom_bytes", 4 * 1024**3)):
                if sample.get(key) is None or int(sample[key]) < floor:
                    raise ValueError(f"runtime {key} below floor")
            remaining = int(sample["projected_remaining_write_bytes"])
            if (sample.get("disk_free_bytes") is None or
                    int(sample["disk_free_bytes"]) < max(1024**3, remaining + 1024**3)):
                raise ValueError("runtime disk reserve or projected remaining writes failed")
            if self.output_root is not None and shutil.disk_usage(self.output_root.parent).free < 1024**3:
                raise ValueError("output volume has less than 1 GiB free")
        except Exception as exc:
            raise TransitionError(f"runtime resource supervisor sample unavailable/failed: {exc}") from exc

    def close(self):
        if self.rows is not None:
            self.rows.close()
        self.env.close()


def create_adapter(*, config: Path, archive: Path, frame_cap: int,
                   artifact_cap: int, wall_cap: float,
                   resource_sample_path: Path | None = None, output_root: Path | None = None) -> Level1DQfDAdapter:
    """Construct Level1Env only when the CLI was explicitly put in replay mode."""
    config_path = Path(config)
    archive_path = Path(archive)
    pinned = {
        "config": (config_path, "f2a28db6efc046b8527fab61b39a21fff2de8e3762eedc4638f0eb8e193f6d7d"),
        "archive": (archive_path, "2cc2fd52ae5eb23d042368f33c108d58d57b539777addd6bb33eedbba63f6103"),
        "route": (ROOT / "training/backward_level1_v1/route.npz", "214b5d56f858a45690779c11441ee3251880a1e8bf3418e564f461fe655b6eed"),
        "route_metadata": (ROOT / "training/backward_level1_v1/route.json", "d4b0cb74e06705671bca8699d692b410d7c06901a8e2098596007d3d918b26d6"),
        "rom": (ROOT / "roms/prince_valiant.nes", "96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5"),
        "reset_389": (ROOT / "training/backward_level1_v1/states/0000389.state.z", "49bbaebd343ad4b19133d7b7788bef9d8868911224eb722a667b6a55fe7c4810"),
        "reset_18000": (ROOT / "training/backward_level1_v1/states/0018000.state.z", "42f20a47179ea9b88222bf2f8cbc64399c9646f54173bd716e1162d2b7e8f479"),
        "reset_17940": (ROOT / "training/backward_level1_v1/states/0017940.state.z", "413603530c3d522ff6deb6faf7c92b093cb7fc4cb2ab6cd777eaf20b36b33eb0"),
    }
    for label, (path, expected) in pinned.items():
        if sha256_file(path) != expected:
            raise TransitionError(f"pinned {label} changed after preflight")
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    observer_enabled = bool(config_data.get("combat_observer_diagnostics", False))
    observer_pin = {}
    if observer_enabled:
        build = ROOT / "instrumentation/kind18_observer_build"
        provenance = json.loads((build / "provenance.json").read_text(encoding="utf-8"))
        observer_path = build / "kind18_observer.dll"
        base_path = ROOT / "instrumentation/recorder_build/recorder_native.dll"
        observer_pin = {"observer_dll_sha256": sha256_file(observer_path),
                        "provenance_sha256": sha256_file(build / "provenance.json"),
                        "base_dll_sha256": sha256_file(base_path),
                        "declared_observer_sha256": provenance.get("observer_dll_sha256"),
                        "declared_base_sha256": provenance.get("base_dll_sha256")}
        if observer_pin["observer_dll_sha256"] != observer_pin["declared_observer_sha256"] or observer_pin["base_dll_sha256"] != observer_pin["declared_base_sha256"]:
            raise TransitionError("observer/base DLL pin differs from on-disk provenance before native construction")
    rows = _ArchiveRows(archive_path)
    capacity = frame_index_capacity(rows.headers, offsets=config_data.get("history_offsets", ()))
    rows.projected_cache_bytes = int(capacity["projected_total_bytes"])
    if not capacity["fits"]:
        rows.close()
        raise TransitionError(f"v2 frame-indexed cache projection {capacity['projected_total_bytes']} exceeds 1 GiB")
    projected_bytes = rows.projected_cache_bytes
    if projected_bytes > artifact_cap:
        rows.close()
        raise TransitionError(f"projected indexed cache {projected_bytes} exceeds artifact cap {artifact_cap}")
    # This import constructs a RecorderNative owner; keep it inside this explicit
    # function so import, help and preflight remain native-free.
    env = None
    try:
        from prince_valiant_rl.backward_level1 import HumanRoute, Level1Env
        route = HumanRoute(directory=ROOT / "training/backward_level1_v1", config=config_data)
        env = Level1Env(config_data, route=route)
        if observer_enabled:
            identity = getattr(env.native, "abi_identity", None)
            if not isinstance(identity, dict) or identity.get("observer_dll_sha256") != observer_pin["observer_dll_sha256"] or identity.get("base_dll_sha256") != observer_pin["base_dll_sha256"]:
                env.close()
                raise TransitionError("runtime loaded observer/base identity differs from preconstruction pins")
        adapter = Level1DQfDAdapter(env, archive_path, rows=rows, frame_cap=frame_cap,
                                    artifact_cap=artifact_cap, wall_cap=wall_cap,
                                    history_offsets=config_data.get("history_offsets", ()),
                                    resource_sample_path=resource_sample_path, output_root=output_root)
        adapter._observer_pin = observer_pin or None
        return adapter
    except Exception as exc:
        try:
            rows.close()
        except Exception as cleanup_exc:
            exc.add_note(f"archive cleanup also failed: {cleanup_exc!r}")
        if env is not None:
            try:
                env.close()
            except Exception as cleanup_exc:
                exc.add_note(f"environment cleanup also failed: {cleanup_exc!r}")
        raise
