"""Offline-safe DQfD transition cache primitives.

Importing this module never imports the emulator or training stack.  A replay
adapter is supplied by the explicit caller and must implement the small
protocol documented by :func:`build_cache`.
"""
from __future__ import annotations

import hashlib
import ctypes
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from .dqfd_receipts import (MAX_JOURNAL_BYTES, MAX_RECORD_BYTES, append_record,
                            read_records, serialize_record)

FORMAT = "level1_dqfd_transition_cache_v2_frame_indexed"
LEGACY_FORMAT = "level1_dqfd_transition_cache_v1"
ACTION_MASKS = (0, 128, 64, 32, 16, 2, 1, 129, 130, 65, 131, 160, 176, 144, 34, 66, 96)
SOURCES = (
    ("original_389", 18273, 389, 0, "human_original_route"),
    ("human18000_noop2", 836, 18000, 2, "human_correction"),
    ("human17940_noop2", 837, 17940, 2, "human_correction"),
    ("ppo_noop1", 838, 17940, 1, "verified_successful_policy_replay"),
    ("ppo_noop16", 815, 17940, 16, "verified_successful_policy_replay"),
)
ROW_COUNT = sum(x[1] for x in SOURCES)
CONTEXT_FRAMES = sum(x[3] for x in SOURCES)
STAGE_CHECKPOINT_ROWS = 256
V3_CACHE_CAP_BYTES = 1024 ** 3
MAX_IMMUTABLE_JSON_BYTES = 64 * 1024
MAX_OUTCOME_RECORD_BYTES = 12 * 1024
MAX_OUTCOME_JOURNAL_BYTES = 272 * 1024 * 1024
MAX_CONTEXT_RECEIPT_BYTES = 16 * 1024
ERROR_PUBLICATION_RESERVE_BYTES = 64 * 1024


def frame_index_capacity(headers: Mapping[str, Mapping[str, Any]], *,
                         offsets: Sequence[int] = (-60, -30, -15, -9, -6, -3, 0)) -> dict[str, int]:
    """Derive a conservative v2 byte bound from validated NPY headers and fixed source spans."""
    metadata = {"frame", "action", "action_mask", "confidence", "changed_action",
                "previous_route_mask", "post_ram_sha256", "provenance_kind", "trajectory_id"}
    obs = {key: value for key, value in headers.items() if key not in metadata}
    valid = obs.get("history_valid")
    if valid is None or valid.get("shape") != [ROW_COUNT, len(offsets)]:
        raise TransitionError("v2 preflight requires the actual seven-slot history_valid header")
    if valid.get("dtype") != "<f4":
        raise TransitionError("v2 preflight history_valid dtype differs from the archived float32 contract")
    temporal = {}
    non_temporal = {}
    for key, value in obs.items():
        shape = list(value.get("shape", []))
        dtype = np.dtype(value.get("dtype"))
        if not shape or shape[0] != ROW_COUNT or dtype.hasobject or value.get("fortran_order"):
            raise TransitionError(f"unsupported source observation header for {key}")
        if key == "history_valid":
            continue
        if shape and shape[0] == ROW_COUNT and len(shape) > 1 and shape[1] == len(offsets):
            if int(value["nbytes"]) % (ROW_COUNT * len(offsets)):
                raise TransitionError(f"temporal observation byte count is not divisible for {key}")
            per_frame = int(value["nbytes"]) // (ROW_COUNT * len(offsets))
            temporal[key] = per_frame
        else:
            non_temporal[key] = int(value["nbytes"])
    if not temporal:
        raise TransitionError("v2 preflight found no validated temporal observation fields")
    per_frame = sum(temporal.values())
    frame_capacity = sum(count + prefix + 1 - min(offsets)
                         for _name, count, _reset, prefix, _kind in SOURCES)
    window_rows = ROW_COUNT + len(SOURCES)
    frame_arrays = frame_capacity * per_frame
    frame_valid = frame_capacity
    references = window_rows * len(offsets) * 8
    validity = window_rows * int(valid["nbytes"]) // ROW_COUNT
    padding = len(SOURCES) * per_frame
    non_temporal_bytes = sum((int(v) // ROW_COUNT) * window_rows for v in non_temporal.values())
    transitions = ROW_COUNT * 4097
    outcomes = (ROW_COUNT + CONTEXT_FRAMES) * MAX_OUTCOME_RECORD_BYTES
    receipt_bounds = progress_receipt_budget()
    if not receipt_bounds["fits"]:
        raise TransitionError("compact progress/context envelope proof exceeds bounded journal caps")
    context_records = int(receipt_bounds["context_journal_bytes_upper_bound"])
    progress = int(receipt_bounds["progress_journal_bytes_upper_bound"])
    resource = 5 * 1024 * 1024
    immutable_and_publication = 2 * 1024 * 1024
    diagnostic_reserve = 128 * 1024 * 1024
    total = (frame_arrays + frame_valid + references + validity + padding + non_temporal_bytes
             + transitions + outcomes + context_records + progress + resource + immutable_and_publication
             + diagnostic_reserve)
    return {"temporal_field_count": len(temporal), "non_temporal_field_count": len(non_temporal),
        "bytes_per_encoded_frame": per_frame, "source_local_frame_capacity": frame_capacity,
        "frame_array_bytes": frame_arrays, "frame_valid_bytes": frame_valid,
        "reference_bytes": references, "history_valid_bytes": validity,
        "padding_bytes_upper_bound": padding, "non_temporal_bytes_upper_bound": non_temporal_bytes,
        "transition_receipt_bytes_upper_bound": transitions,
        "outcome_receipt_bytes_upper_bound": outcomes,
        "outcome_receipt_max_bytes": MAX_OUTCOME_RECORD_BYTES,
        "outcome_journal_cap_bytes": MAX_OUTCOME_JOURNAL_BYTES,
        "context_journal_bytes_upper_bound": context_records,
        "progress_journal_bytes_upper_bound": progress,
        "resource_journal_bytes_upper_bound": resource,
        "progress_checkpoint_count": int(receipt_bounds["progress_checkpoint_count"]),
        "max_progress_record_bytes": int(receipt_bounds["max_progress_record_bytes"]),
        "max_context_record_bytes": int(receipt_bounds["max_context_record_bytes"]),
        "progress_and_resource_journals": progress + resource,
        "manifest_error_ready_and_publication_bytes": immutable_and_publication,
        "diagnostic_reserve_bytes": diagnostic_reserve,
        "projected_total_bytes": total, "cache_cap_bytes": V3_CACHE_CAP_BYTES,
        "fits": total <= V3_CACHE_CAP_BYTES}


def progress_receipt_budget(*, rows: int = ROW_COUNT, contexts: int = CONTEXT_FRAMES,
                            sources: int = len(SOURCES), frame_cap: int = 79960) -> dict[str, int | bool]:
    """Bound actual progress envelopes/checkpoint count without cumulative diagnostics."""
    summaries = [{"name": "source_identity_000000", "rows": rows, "source_id": i,
        "source_kind": "verified_successful_policy_replay", "reset_frame": 18000,
        "start_frame": 18016, "context_frames": contexts, "reset_frames": 0,
        "context_ref": {"path": "context_receipts.jsonl", "first_sequence": 21,
            "last_sequence": 21, "count": 16, "sha256": "f" * 64}}
        for i in range(sources)]
    ledger = {phase: {key: frame_cap for key in (
        "planned", "attempted", "completed", "uncertain", "planned_actions",
        "attempted_actions", "completed_actions", "uncertain_actions")}
        for phase in ("reset", "context", "labeled")}
    progress_payload = {"format": FORMAT, "attempt_id": "f" * 32, "status": "finalizing",
        "rows_written": rows, "frame_cap": frame_cap, "native_frames": frame_cap,
        "completed_sources": sources, "source_summaries": summaries,
        "context_receipts": {"path": "context_receipts.jsonl", "count": contexts,
            "last_sequence": contexts, "last_sha256": "f" * 64},
        "current_trajectory": "source_identity_000000", "current_row": rows,
        "current_context_index": contexts, "current_context_range": [1, contexts],
        "frame_ledger": ledger, "elapsed_seconds": 1273.0}
    timestamp = "9999-12-31T23:59:59.999999+00:00"
    progress_record = len(serialize_record(progress_payload, attempt_id="f" * 32,
        sequence=32768, written_at_utc=timestamp)[1])
    checkpoint_count = 1 + contexts + rows // STAGE_CHECKPOINT_ROWS + sources + 1
    context_payload = {"source": "source_identity_000000", "index": contexts,
        "action_mask": 0, "duration": 1, "reward": -1e308,
        "components": {f"reward_component_{i:02d}": -1e308 for i in range(64)},
        "availability": {f"reward_component_{i:02d}": bool(i % 2) for i in range(64)},
        "score_state": {"diagnostic": "x" * 4096,
            "coverage": {f"score_{i:02d}": {"available": bool(i % 2), "count": frame_cap,
                "points": -1e308} for i in range(64)}},
        "reward_state": {"tracker_frontiers": [frame_cap] * 64,
            "tracker_visited": list(range(64)), "tracker_objectives": list(range(64)),
            "reward_totals": {f"component_{i:02d}": -1e308 for i in range(64)},
            "discounted_reward_totals": {f"component_{i:02d}": -1e308 for i in range(64)}},
        "end_reason": ""}
    context_record = len(serialize_record(context_payload, attempt_id="f" * 32,
        sequence=contexts, written_at_utc=timestamp)[1])
    progress_journal = progress_record * checkpoint_count
    context_journal = context_record * contexts
    return {"progress_checkpoint_count": checkpoint_count,
        "max_progress_record_bytes": progress_record,
        "progress_journal_bytes_upper_bound": progress_journal,
        "max_context_record_bytes": context_record,
        "context_journal_bytes_upper_bound": context_journal,
        "progress_journal_cap_bytes": MAX_JOURNAL_BYTES,
        "context_journal_cap_bytes": MAX_JOURNAL_BYTES,
        "record_cap_bytes": MAX_RECORD_BYTES,
        "fits": (progress_record <= MAX_RECORD_BYTES and context_record <= MAX_RECORD_BYTES
            and progress_journal <= MAX_JOURNAL_BYTES and context_journal <= MAX_JOURNAL_BYTES)}


class TransitionError(RuntimeError):
    """Bounded, receipt-ready replay or cache failure."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest()[:16]}


def compare_window(expected: Mapping[str, Any], actual: Mapping[str, Any], *, trajectory: str, row: int, frame: int) -> None:
    if set(expected) != set(actual):
        raise TransitionError(f"{trajectory}/{row}@{frame}: field set mismatch expected={sorted(expected)} actual={sorted(actual)}")
    for key in expected:
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right, equal_nan=False):
            raise TransitionError(f"{trajectory}/{row}@{frame}: field {key} mismatch expected={_summary(left)} actual={_summary(right)}")


def lossless_cast(values: Any, dtype: Any) -> np.ndarray:
    """Cast only when every value round-trips exactly (including NaNs)."""
    source = np.asarray(values)
    target_dtype = np.dtype(dtype)
    converted = source.astype(target_dtype)
    restored = converted.astype(source.dtype)
    if source.dtype.kind in "fc":
        equal = np.array_equal(source, restored, equal_nan=True)
    else:
        equal = np.array_equal(source, restored)
    if not equal:
        raise TransitionError(f"lossy or out-of-range storage cast {source.dtype} -> {target_dtype}")
    return converted


def boundary(*, natural_terminal: bool, time_limit: bool, end_reason: str, game_terminal: bool) -> dict[str, Any]:
    if natural_terminal and time_limit:
        raise TransitionError("boundary cannot be both natural terminal and time limit")
    if natural_terminal != bool(game_terminal):
        raise TransitionError("game_terminal does not agree with natural terminal")
    if natural_terminal and not end_reason:
        raise TransitionError("natural terminal requires end_reason")
    return {"natural_terminal": bool(natural_terminal), "time_limit": bool(time_limit),
            "game_terminal": bool(game_terminal), "end_reason": str(end_reason),
            "bootstrap": not bool(natural_terminal)}


class FrameLedger:
    def __init__(self, frame_cap: int):
        if frame_cap < 0:
            raise ValueError("frame cap must be nonnegative")
        self.cap, self.actual, self.planned = int(frame_cap), 0, 0

    def plan(self, frames: int) -> None:
        if frames < 0 or self.actual + self.planned + frames > self.cap:
            raise TransitionError(f"frame cap exceeded before advance: actual={self.actual} planned={self.planned} request={frames} cap={self.cap}")
        self.planned += int(frames)

    def commit(self, frames: int) -> None:
        if frames < 0 or frames > self.planned:
            raise TransitionError("actual frame advance exceeds planned ledger")
        self.planned -= int(frames)
        self.actual += int(frames)


def _atomic_json(path: Path, obj: Any) -> None:
    """One-time publication to an absent destination; never replace a hot file."""
    if path.exists():
        raise TransitionError(f"one-time publication destination is occupied: {path}")
    tmp = path.with_name(path.name + ".publish-" + uuid.uuid4().hex + ".tmp")
    data = (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > MAX_IMMUTABLE_JSON_BYTES:
        raise TransitionError("immutable JSON receipt exceeds 64 KiB bound")
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _guard(adapter: Any, ledger: FrameLedger, *, artifact_cap: int, wall_cap: float, start: float, root: Path,
           projected_write_bytes: int = 0) -> None:
    if time.monotonic() - start > wall_cap:
        raise TransitionError("wall cap reached during bounded cache operation")
    used = int(getattr(adapter, "artifact_bytes_written", 0))
    reserved = int(getattr(adapter, "growth_reserved_bytes", 0))
    reserve = max(int(getattr(adapter, "estimated_next_write_bytes", 65536)), int(projected_write_bytes))
    projected = int(getattr(adapter, "estimated_total_artifact_bytes", 0))
    if projected > artifact_cap:
        raise TransitionError(f"projected artifacts {projected} exceed cap {artifact_cap} before advance; accounting={_accounting_snapshot(adapter, root, artifact_cap)}")
    diagnostic = int(getattr(adapter, "diagnostic_reserve_bytes", 0))
    if used + reserved + diagnostic + reserve > artifact_cap:
        raise TransitionError(f"artifact cap reached before bounded cache operation; accounting={_accounting_snapshot(adapter, root, artifact_cap)}")
    check = getattr(adapter, "check_runtime_guards", None)
    if check:
        check()
    if ledger.actual + ledger.planned > ledger.cap:
        raise TransitionError("native frame cap reached before next action")


def _accounting_snapshot(adapter: Any, root: Path, cap: int) -> dict[str, int]:
    return {"logical_allocated_bytes": int(getattr(adapter, "artifact_bytes_written", 0)),
        "outstanding_growth_reservations": int(getattr(adapter, "growth_reserved_bytes", 0)),
        "diagnostic_reserve_bytes": int(getattr(adapter, "diagnostic_reserve_bytes", 0)),
        "estimated_file_bytes": int(getattr(adapter, "estimated_total_artifact_bytes", 0)),
        "last_reconciled_file_bytes": int(getattr(adapter, "last_reconciled_file_bytes", 0)),
        "actual_allocated_disk_bytes": getattr(adapter, "actual_allocated_disk_bytes", None),
        "actual_write_traffic_bytes": int(getattr(adapter, "actual_write_traffic_bytes", 0)),
        "write_traffic_measurement": "application-level array/receipt payloads submitted; OS/device physical transfer unavailable",
        "physical_device_write_traffic_bytes": None,
        "cache_cap_bytes": int(cap)}


def _reserve_growth(adapter: Any, amount: int) -> None:
    adapter.growth_reserved_bytes = int(getattr(adapter, "growth_reserved_bytes", 0)) + int(amount)


def _commit_growth(adapter: Any, reserved: int, actual: int) -> None:
    adapter.growth_reserved_bytes = max(0, int(getattr(adapter, "growth_reserved_bytes", 0)) - int(reserved))
    adapter.artifact_bytes_written += int(actual)
    adapter.actual_write_traffic_bytes += int(actual)


def _reconcile_files(adapter: Any, root: Path) -> int:
    files = [p for p in root.rglob("*") if p.is_file()]
    actual = sum(p.stat().st_size for p in files)
    allocated = 0
    if os.name == "nt":
        try:
            high = ctypes.c_ulong()
            get_size = ctypes.windll.kernel32.GetCompressedFileSizeW
            get_size.restype = ctypes.c_ulong
            for path in files:
                high.value = 0
                low = int(get_size(str(path), ctypes.byref(high)))
                if low == 0xFFFFFFFF and ctypes.get_last_error():
                    raise ctypes.WinError(ctypes.get_last_error())
                allocated += (int(high.value) << 32) | low
            adapter.actual_allocated_disk_bytes = allocated
        except Exception:
            adapter.actual_allocated_disk_bytes = None
    else:
        adapter.actual_allocated_disk_bytes = None
    adapter.last_reconciled_file_bytes = actual
    logical = int(getattr(adapter, "artifact_bytes_written", 0))
    if actual > logical:
        adapter.artifact_bytes_written += actual - logical
    return actual


def _append_outcome(path: Path, payload: Mapping[str, Any], *, attempt_id: str,
                    sequence: int) -> int:
    body = {"attempt_id": str(attempt_id), "sequence": int(sequence), "payload": dict(payload)}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    envelope = {**body, "sha256": hashlib.sha256(canonical).hexdigest()}
    data = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    if len(data) > MAX_OUTCOME_RECORD_BYTES:
        raise TransitionError(f"bounded outcome receipt exceeds {MAX_OUTCOME_RECORD_BYTES} bytes")
    if (path.stat().st_size if path.exists() else 0) + len(data) > MAX_OUTCOME_JOURNAL_BYTES:
        raise TransitionError("outcome journal reached its bounded size")
    with path.open("ab") as stream:
        if stream.write(data) != len(data):
            raise OSError("short outcome journal write")
        stream.flush()
        os.fsync(stream.fileno())
    return len(data)


def _uncertain_frames(adapter: Any) -> int:
    counts = getattr(adapter, "frame_ledger", {})
    return sum(int(row.get("uncertain", 0)) for row in counts.values()) if isinstance(counts, Mapping) else 0


def _completed_frames(adapter: Any) -> int:
    counts = getattr(adapter, "frame_ledger", {})
    return sum(int(row.get("completed", 0)) for row in counts.values()) if isinstance(counts, Mapping) else 0


def _validate_outcomes(path: Path, *, attempt_id: str, expected_count: int) -> int:
    count, total = 0, 0
    with Path(path).open("rb") as stream:
        for line in stream:
            total += len(line)
            if len(line) > MAX_OUTCOME_RECORD_BYTES or not line.endswith(b"\n"):
                raise TransitionError("outcome receipt exceeds its bounded record or framing contract")
            envelope = json.loads(line)
            body = {key: value for key, value in envelope.items() if key != "sha256"}
            if (envelope.get("attempt_id") != attempt_id or int(envelope.get("sequence", -1)) != count + 1
                    or envelope.get("sha256") != hashlib.sha256(json.dumps(body, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()):
                raise TransitionError("outcome receipt identity, sequence, or checksum mismatch")
            count += 1
    if count != int(expected_count) or total > MAX_OUTCOME_JOURNAL_BYTES:
        raise TransitionError("outcome receipt count or bounded journal size mismatch")
    return total


def build_cache(*, trajectories: Sequence[Mapping[str, Any]], adapter: Any, output: Path,
                frame_cap: int, artifact_cap: int, wall_cap: float,
                lineage: Mapping[str, Any] | None = None,
                identity_receipt: Mapping[str, Any] | None = None,
                progress_path: Path | None = None,
                attempt_id: str | None = None) -> dict[str, Any]:
    """Build a one-shot cache using an injected source replay adapter.

    Each trajectory item contains name, expected rows, context_frames, source_kind
    and an iterable of row dictionaries. The adapter has reset(source),
    observation(), step(mask), and close() methods. reset/step return actual
    advanced native frames; step returns observation, info, RAM bytes and reward.
    """
    output = Path(output)
    if output.exists():
        raise TransitionError(f"output root is occupied: {output}")
    progress_path = Path(progress_path) if progress_path is not None else output / "progress.jsonl"
    if progress_path.exists():
        raise TransitionError(f"progress journal path is occupied: {progress_path}")
    output.mkdir(parents=True)
    start, ledger = time.monotonic(), FrameLedger(frame_cap)
    attempt_id = str(attempt_id or uuid.uuid4().hex)
    progress_seq = 0
    context_path = output / "context_receipts.jsonl"
    outcome_path = output / "outcomes.jsonl"
    outcome_seq = 0
    context_seq = 0
    context_hashes: dict[str, list[str]] = {}
    stage: dict[str, Any] = {"format": FORMAT, "attempt_id": attempt_id, "status": "partial", "rows_written": 0,
                             "frame_cap": frame_cap, "native_frames": 0, "completed_sources": 0,
                             "source_summaries": [],
                             "context_receipts": {"path": context_path.name, "count": 0}}
    adapter.growth_reserved_bytes = 0
    adapter.actual_write_traffic_bytes = 0
    adapter.diagnostic_reserve_bytes = ERROR_PUBLICATION_RESERVE_BYTES
    adapter.last_reconciled_file_bytes = 0
    adapter.actual_allocated_disk_bytes = None
    def checkpoint() -> None:
        nonlocal progress_seq
        progress_seq += 1
        envelope = append_record(progress_path, stage, attempt_id=attempt_id, sequence=progress_seq)
        record_size = len(json.dumps(envelope, sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")) + 1
        adapter.artifact_bytes_written += record_size
        adapter.actual_write_traffic_bytes += record_size
    metadata = None
    store = None
    try:
        with context_path.open("xb") as context_stream:
            context_stream.flush()
            os.fsync(context_stream.fileno())
        adapter.artifact_bytes_written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start,
               root=output, projected_write_bytes=MAX_IMMUTABLE_JSON_BYTES)
        _reserve_growth(adapter, MAX_IMMUTABLE_JSON_BYTES)
        identity = dict(identity_receipt or {"attempt_id": attempt_id, "lineage": dict(lineage or {})})
        _atomic_json(output / "identity.json", identity)
        identity_size = (output / "identity.json").stat().st_size
        _commit_growth(adapter, MAX_IMMUTABLE_JSON_BYTES, identity_size)
        stage["identity_ref"] = {"path": "identity.json", "sha256": sha256_file(output / "identity.json")}
        checkpoint()
        metadata = (output / "transitions.jsonl").open("x", encoding="utf-8")
        windows_dir = output / "windows"
        windows_dir.mkdir()
        def before_allocate(amount: int) -> None:
            _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start,
                   root=output, projected_write_bytes=amount)
            adapter.artifact_bytes_written += int(amount)
            adapter.actual_write_traffic_bytes += int(amount)
        offsets = tuple(getattr(adapter, "history_offsets", (-6, -5, -4, -3, -2, -1, 0)))
        if len(offsets) != 7 or offsets[-1] != 0 or tuple(sorted(set(offsets))) != offsets:
            raise TransitionError("frame-indexed cache requires the verified seven-slot ascending history offsets")
        store = _FrameIndexedWindowStore(windows_dir, offsets=offsets,
            temporal_fields=getattr(adapter, "temporal_fields", None), before_allocate=before_allocate)
        for source in trajectories:
            name = str(source["name"])
            rows = iter(source["rows"])
            expected_count = int(source["expected_rows"])
            prefix = int(source.get("context_frames", 0))
            reset_frame = int(source.get("reset_frame", int(source.get("start_frame", 0)) - prefix))
            store.configure(name, expected_count, reset_frame=reset_frame, context_frames=prefix,
                            source_id=int(source.get("source_id", -1)))
            # All initialization work must be declared by the adapter before reset.
            _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output)
            planned = int(adapter.planned_reset_frames(source))
            ledger.plan(planned)
            reset = adapter.reset(source)
            reset_frames = int(reset.get("native_frames", 0))
            ledger.commit(reset_frames)
            if reset_frames != planned:
                raise TransitionError(f"{name}: reset advanced {reset_frames}, planned {planned}")
            context_first = context_seq + 1
            context_hashes[name] = []
            for ci in range(prefix):
                _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
                       projected_write_bytes=MAX_OUTCOME_RECORD_BYTES + MAX_CONTEXT_RECEIPT_BYTES)
                _reserve_growth(adapter, MAX_OUTCOME_RECORD_BYTES + MAX_CONTEXT_RECEIPT_BYTES)
                ledger.plan(1)
                adapter._context_dispatch = True
                try:
                    result = adapter.step(0)
                finally:
                    adapter._context_dispatch = False
                duration_raw = result.get("duration")
                next_outcome_seq = outcome_seq + 1
                outcome = {"source": name, "local_row": None, "context_index": ci,
                    "executed_mask": result.get("executed_mask"), "duration": duration_raw,
                    "native_completed": _completed_frames(adapter), "native_uncertain": _uncertain_frames(adapter),
                    "reward": result.get("reward"), "components": result.get("components", {}),
                    "availability": result.get("availability", {}), "score_coverage": result.get("score_state", {}),
                    "post_ram_sha256": hashlib.sha256(bytes(result["ram"])).hexdigest() if result.get("ram") is not None else None,
                    "parity_validated": False, "accepted_demonstration_row": False}
                outcome_size = _append_outcome(outcome_path, outcome, attempt_id=attempt_id, sequence=next_outcome_seq)
                outcome_seq = next_outcome_seq
                _commit_growth(adapter, MAX_OUTCOME_RECORD_BYTES, outcome_size)
                duration = int(duration_raw)
                ledger.commit(duration)
                _validate_measurements(result, f"{name}/context{ci}")
                if int(result.get("executed_mask", -1)) != 0:
                    raise TransitionError(f"{name}: context NOOP executed-mask mismatch")
                context_receipt = {"source": name, "index": ci, "action_mask": 0, "duration": duration,
                                         "reward": float(result["reward"]), "components": result.get("components", {}),
                                         "availability": result.get("availability", {}),
                                         "score_state": result.get("score_state", {}),
                                         "reward_state": result.get("reward_state", {}),
                                         "end_reason": result.get("info", {}).get("end_reason", "")}
                context_seq += 1
                envelope = append_record(context_path, context_receipt, attempt_id=attempt_id,
                                         sequence=context_seq)
                context_bytes = len(json.dumps(envelope, sort_keys=True, separators=(",", ":"),
                    allow_nan=False).encode("utf-8")) + 1
                _commit_growth(adapter, MAX_CONTEXT_RECEIPT_BYTES, context_bytes)
                context_hashes[name].append(str(envelope["sha256"]))
                stage["context_receipts"] = {"path": context_path.name, "count": context_seq,
                    "last_sequence": context_seq, "last_sha256": str(envelope["sha256"])}
                stage.update(native_frames=ledger.actual, current_trajectory=name,
                             current_row=0, current_context_index=ci,
                             current_context_range=[context_first, context_seq])
                checkpoint()
                _reconcile_files(adapter, output)
            count = 0
            pending = None
            while True:
                try:
                    row = next(rows)
                except StopIteration:
                    break
                if count >= expected_count:
                    raise TransitionError(f"{name}: extra row after expected count")
                frame = int(row["frame"])
                if row.get("source_kind") != source.get("source_kind"):
                    raise TransitionError(f"{name}/{count}@{frame}: source-kind identity mismatch")
                if "source_id" in source and int(row.get("source_id", -1)) != int(source["source_id"]):
                    raise TransitionError(f"{name}/{count}@{frame}: source trajectory identity mismatch")
                if "start_frame" in source and frame != int(source["start_frame"]) + count:
                    raise TransitionError(f"{name}/{count}@{frame}: missing or reordered source frame")
                expected_pre = row["pre"]
                actual_pre = adapter.observation()
                compare_window(expected_pre, actual_pre, trajectory=name, row=count, frame=frame)
                action = int(row["action"])
                raw_mask = int(row["action_mask"])
                if action < 0 or action >= len(ACTION_MASKS) or ACTION_MASKS[action] != raw_mask:
                    raise TransitionError(f"{name}/{count}@{frame}: action/index-mask mismatch index={action} mask={raw_mask}")
                _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
                       projected_write_bytes=MAX_OUTCOME_RECORD_BYTES + 4097)
                _reserve_growth(adapter, MAX_OUTCOME_RECORD_BYTES + 4097)
                store.prepare(name, expected_count, expected_pre)
                ledger.plan(1)
                result = adapter.step(raw_mask)
                duration_raw = result.get("duration")
                next_outcome_seq = outcome_seq + 1
                raw_ram = result.get("ram")
                outcome = {"source": name, "local_row": count, "context_index": None,
                    "executed_mask": result.get("executed_mask"), "duration": duration_raw,
                    "native_completed": _completed_frames(adapter), "native_uncertain": _uncertain_frames(adapter),
                    "reward": result.get("reward"), "components": result.get("components", {}),
                    "availability": result.get("availability", {}), "score_coverage": result.get("score_state", {}),
                    "post_ram_sha256": hashlib.sha256(bytes(raw_ram)).hexdigest() if raw_ram is not None else None,
                    "parity_validated": False, "accepted_demonstration_row": False}
                outcome_size = _append_outcome(outcome_path, outcome, attempt_id=attempt_id, sequence=next_outcome_seq)
                outcome_seq = next_outcome_seq
                _commit_growth(adapter, MAX_OUTCOME_RECORD_BYTES, outcome_size)
                duration = int(duration_raw)
                ledger.commit(duration)
                if duration < 1 or duration > 1:
                    raise TransitionError(f"{name}/{count}@{frame}: frame_skip=1 but duration={duration}")
                if int(result.get("executed_mask", -1)) != raw_mask:
                    raise TransitionError(f"{name}/{count}@{frame}: actual executed mask mismatch expected={raw_mask} actual={result.get('executed_mask')}")
                _validate_measurements(result, f"{name}/{count}@{frame}")
                if result.get("info", {}).get("autoreset", False):
                    if not result.get("terminal_observation_verified", False) or "terminal_observation" not in result:
                        raise TransitionError(f"{name}/{count}@{frame}: autoreset terminal observation is not verified")
                    post = result["terminal_observation"]
                else:
                    post = result["observation"]
                digest = hashlib.sha256(bytes(result["ram"])).hexdigest()
                if digest != str(row["post_ram_sha256"]):
                    raise TransitionError(f"{name}/{count}@{frame}: post-RAM digest mismatch expected={row['post_ram_sha256']} actual={digest}")
                if not math.isfinite(float(result["reward"])):
                    raise TransitionError(f"{name}/{count}@{frame}: non-finite reward")
                if result.get("invalid_bcd", False):
                    raise TransitionError(f"{name}/{count}@{frame}: invalid BCD is fatal")
                info = result.get("info", {})
                b = boundary(natural_terminal=bool(info.get("natural_terminal", False)),
                             time_limit=bool(info.get("time_limit", False)),
                             end_reason=str(info.get("end_reason", "")),
                             game_terminal=bool(info.get("game_terminal", False)))
                actual_bootstrap = info.get("bootstrap_valid")
                if not isinstance(actual_bootstrap, (bool, np.bool_)) or bool(actual_bootstrap) != b["bootstrap"]:
                    raise TransitionError(f"{name}/{count}@{frame}: actual bootstrap eligibility missing or inconsistent")
                if "transition_frames" in info and int(info["transition_frames"]) != duration:
                    raise TransitionError(f"{name}/{count}@{frame}: duration differs from environment info")
                if count + 1 < expected_count:
                    if b["natural_terminal"] or b["time_limit"]:
                        raise TransitionError(f"{name}/{count}@{frame}: episode ended before its saved final row")
                    compare_window(rows_peek := row["next_pre"], post, trajectory=name, row=count, frame=frame)
                    next_ref = {"kind": "pre", "row": count + 1}
                else:
                    if not (b["natural_terminal"] or b["time_limit"]):
                        raise TransitionError(f"{name}/{count}@{frame}: final row has no natural or time-limit boundary")
                    expected_reason = source.get("expected_end_reason")
                    if expected_reason and b["end_reason"] != expected_reason:
                        raise TransitionError(f"{name}/{count}@{frame}: wrong final end_reason expected={expected_reason} actual={b['end_reason']}")
                    if info.get("autoreset", False) and not result.get("terminal_observation_verified", False):
                        raise TransitionError(f"{name}/{count}@{frame}: autoreset final state not verified")
                    next_ref = {"kind": "final_post", "trajectory": name, "row": count}
                adapter.actual_write_traffic_bytes += store.write(name, "pre", count,
                    expected_count, expected_pre, current_frame=frame)
                if count + 1 == expected_count:
                    adapter.actual_write_traffic_bytes += store.write(name, "final_post", 0,
                        1, post, current_frame=frame + 1)
                receipt = {"trajectory": name, "local_row": count, "frame": frame, "action": action,
                           "source_id": int(row.get("source_id", source.get("source_id", -1))),
                           "raw_mask": raw_mask, "source_kind": str(row["source_kind"]), "duration": duration,
                           "reward": float(result["reward"]), "components": result.get("components", {}),
                           "availability": result.get("availability", {}), "score_state": result.get("score_state", {}),
                           "reward_state": result.get("reward_state", {}),
                           "post_ram_sha256": digest, "next_ref": next_ref,
                           "bootstrap_valid": bool(actual_bootstrap), **b}
                encoded = json.dumps(receipt, allow_nan=False, sort_keys=True)
                if len(encoded.encode("utf-8")) > 4096:
                    raise TransitionError(f"{name}/{count}@{frame}: row receipt exceeds 4 KiB bound")
                _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start,
                       root=output, projected_write_bytes=len(encoded.encode("utf-8")) + 1)
                metadata.write(encoded + "\n")
                metadata.flush()
                _commit_growth(adapter, 4097, len(encoded.encode("utf-8")) + 1)
                count += 1
                stage.update(rows_written=stage["rows_written"] + 1, native_frames=ledger.actual,
                             current_trajectory=name, current_row=count)
                if stage["rows_written"] % STAGE_CHECKPOINT_ROWS == 0:
                    _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start,
                           root=output, projected_write_bytes=65536)
                    checkpoint()
                    _reconcile_files(adapter, output)
            if count != expected_count:
                raise TransitionError(f"{name}: missing rows expected={expected_count} actual={count}")
            stage["completed_sources"] += 1
            source_summary = {"name": name, "rows": count,
                "source_id": int(source.get("source_id", -1)),
                "source_kind": str(source.get("source_kind", "")),
                "reset_frame": reset_frame,
                "start_frame": int(source.get("start_frame", reset_frame + prefix)),
                "context_frames": prefix, "reset_frames": reset_frames,
                "context_ref": {"path": context_path.name, "first_sequence": context_first,
                    "last_sequence": context_seq, "count": prefix,
                    "sha256": hashlib.sha256("".join(context_hashes[name]).encode("ascii")).hexdigest()}}
            stage["source_summaries"].append(source_summary)
            stage["last_completed_source"] = source_summary
            stage.pop("current_context_index", None)
            stage.pop("current_context_range", None)
            checkpoint()
        metadata.close()
        store.flush()
        _reconcile_files(adapter, output)
        if sum(int(x["expected_rows"]) for x in trajectories) != stage["rows_written"]:
            raise TransitionError("global source row count mismatch")
        _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
               projected_write_bytes=0)
        store.verify_readback(guard=lambda amount=0: _guard(adapter, ledger, artifact_cap=artifact_cap,
            wall_cap=wall_cap, start=start, root=output, projected_write_bytes=amount))
        receipt_rows = 0
        receipt_sources = []
        local_counts: dict[str, int] = {}
        required_receipt_fields = {"trajectory", "local_row", "frame", "action", "raw_mask", "source_kind",
            "duration", "reward", "components", "availability", "score_state", "reward_state",
            "post_ram_sha256", "next_ref", "natural_terminal", "time_limit", "game_terminal", "end_reason",
            "bootstrap", "bootstrap_valid"}
        with (output / "transitions.jsonl").open("r", encoding="utf-8") as stream:
            for line in stream:
                if len(line.encode("utf-8")) > 4097:
                    raise TransitionError("receipt readback exceeds per-row bound")
                record = json.loads(line)
                if not required_receipt_fields.issubset(record):
                    raise TransitionError("receipt readback has incomplete transition field coverage")
                owner = str(record["trajectory"])
                local = int(record["local_row"])
                if local != local_counts.get(owner, 0):
                    raise TransitionError(f"receipt readback order mismatch for {owner}: {local}")
                local_counts[owner] = local + 1
                if not receipt_sources or receipt_sources[-1] != owner:
                    receipt_sources.append(owner)
                receipt_rows += 1
        if receipt_rows != stage["rows_written"]:
            raise TransitionError("receipt readback row count mismatch")
        outcome_bytes = _validate_outcomes(outcome_path, attempt_id=attempt_id,
            expected_count=int(stage["rows_written"]) + context_seq)
        if outcome_bytes != outcome_path.stat().st_size:
            raise TransitionError("outcome journal changed during bounded verification")
        expected_counts = {str(x["name"]): int(x["rows"]) for x in stage["source_summaries"]}
        if local_counts != expected_counts or receipt_sources != list(expected_counts):
            raise TransitionError("receipt readback source order/counts do not match completed source ledger")
        store.validate_layout(expected_counts)
        store.flush(close=True)
        stage.update(status="finalizing", native_frames=ledger.actual, elapsed_seconds=round(time.monotonic() - start, 3))
        checkpoint()
        _reconcile_files(adapter, output)
        close = getattr(adapter, "close", None)
        if close:
            adapter._closed_by_builder = True
            close()
        _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
               projected_write_bytes=256 * 1024)
        hashes = {}
        for p in sorted(output.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output)
                hashes[p.relative_to(output).as_posix()] = sha256_file(p)
        manifest = {"format": FORMAT, "status": "ready", "attempt_id": attempt_id,
                    "context_receipts": {"path": context_path.name, "count": context_seq},
                    "outcome_receipts": {"path": outcome_path.name, "count": outcome_seq,
                        "max_record_bytes": MAX_OUTCOME_RECORD_BYTES,
                        "journal_cap_bytes": MAX_OUTCOME_JOURNAL_BYTES},
                    "identity_ref": stage.get("identity_ref"),
                    "lineage": dict(lineage or {}),
                    "rows": stage["rows_written"], "native_frames": ledger.actual,
                    "sources": stage["source_summaries"], "window_schema": store.schema,
                    "frame_index": store.index_manifest(),
                    "history_offsets": list(offsets),
                    "following_reference_contract": "interior rows reference same-trajectory next pre row after exact equality; each final row references its own final_post index 0",
                    "files": hashes, "reward_note": "Rewards are newly generated under the pinned replay config; historical per-row reward receipts are absent."}
        manifest["accounting_at_manifest_publication"] = _accounting_snapshot(adapter, output, artifact_cap)
        _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
               projected_write_bytes=256 * 1024)
        _reserve_growth(adapter, 256 * 1024)
        _atomic_json(output / "manifest.json", manifest)
        manifest_size = (output / "manifest.json").stat().st_size
        _commit_growth(adapter, 256 * 1024, manifest_size)
        _guard(adapter, ledger, artifact_cap=artifact_cap, wall_cap=wall_cap, start=start, root=output,
               projected_write_bytes=4096)
        _reserve_growth(adapter, 4096)
        manifest_hash = sha256_file(output / "manifest.json")
        with TransitionCache(output, unpublished=True):
            pass
        _atomic_json(output / "READY", {"manifest_sha256": manifest_hash})
        ready_size = (output / "READY").stat().st_size
        _commit_growth(adapter, 4096, ready_size)
        _reconcile_files(adapter, output)
        manifest["accounting_after_ready"] = _accounting_snapshot(adapter, output, artifact_cap)
        return manifest
    except Exception as exc:
        try:
            _reconcile_files(adapter, output)
        except Exception as accounting_exc:
            exc.add_note(f"bounded filesystem accounting reconciliation failed: {accounting_exc!r}")
        if metadata is not None:
            try:
                metadata.close()
            except Exception as cleanup_exc:
                stage["metadata_close_error"] = str(cleanup_exc)[:500]
        if store is not None:
            try:
                store.flush(close=True)
            except Exception as cleanup_exc:
                stage["store_close_error"] = str(cleanup_exc)[:500]
        close = getattr(adapter, "close", None)
        if close and not getattr(adapter, "_closed_by_builder", False):
            adapter._closed_by_builder = True
            try:
                close()
            except Exception as cleanup_exc:
                stage["adapter_cleanup_error"] = str(cleanup_exc)[:500]
        counts = getattr(adapter, "frame_ledger", {})
        completed = sum(int(x.get("completed", 0)) for x in counts.values()) if isinstance(counts, Mapping) else None
        uncertain = sum(int(x.get("uncertain", 0)) for x in counts.values()) if isinstance(counts, Mapping) else None
        stage.update(status="error", native_frames=(completed if completed is not None else ledger.actual),
                     frame_ledger=counts, uncertain_native_frames=uncertain,
                     failure=str(exc)[:1000], elapsed_seconds=round(time.monotonic() - start, 3))
        try:
            _atomic_json(output / "error.json", {"attempt_id": attempt_id,
                "format": FORMAT, "status": "error", "rows_written": stage.get("rows_written", 0),
                "native_frames": stage.get("native_frames", ledger.actual),
                "frame_ledger": stage.get("frame_ledger", {}),
                "uncertain_native_frames": stage.get("uncertain_native_frames"),
                "completed_sources": stage.get("completed_sources", 0),
                "source_summaries": stage.get("source_summaries", []),
                "context_receipts": stage.get("context_receipts"),
                "outcome_receipts": {"path": outcome_path.name, "persisted": outcome_seq,
                    "size_bytes": outcome_path.stat().st_size if outcome_path.exists() else 0,
                    "outcome_gap_completed_advances": max(0, sum(int(x.get("completed", 0))
                        for k, x in counts.items() if k != "reset") - outcome_seq),
                    "durability_unconfirmed": max(0, sum(int(x.get("completed", 0))
                        for k, x in counts.items() if k != "reset") - outcome_seq) > 0
                        if isinstance(counts, Mapping) else None},
                "accepted_transition_rows": int(stage.get("rows_written", 0)),
                "identity_ref": stage.get("identity_ref"),
                "accounting": _accounting_snapshot(adapter, output, artifact_cap),
                "current_trajectory": stage.get("current_trajectory"),
                "current_row": stage.get("current_row"),
                "current_context_index": stage.get("current_context_index"),
                "current_context_range": stage.get("current_context_range"),
                "failure": str(exc)[:1000],
                "elapsed_seconds": round(time.monotonic() - start, 3)})
            error_size = (output / "error.json").stat().st_size
            adapter.artifact_bytes_written += error_size
            adapter.actual_write_traffic_bytes += error_size
        except Exception as receipt_exc:
            exc.add_note(f"failure journal write also failed: {receipt_exc!r}")
        raise
    finally:
        close = getattr(adapter, "close", None)
        if close and not getattr(adapter, "_closed_by_builder", False):
            try:
                close()
                adapter._closed_by_builder = True
            except Exception as cleanup_exc:
                if "exc" in locals():
                    exc.add_note(f"adapter cleanup also failed: {cleanup_exc!r}")
                else:
                    raise


class _WindowStore:
    """Trajectory-indexed NPY memmaps; fields retain source dtype and shape."""
    def __init__(self, root: Path, before_allocate: Callable[[int], None] | None = None):
        self.root = root
        self.before_allocate = before_allocate
        self.maps: dict[tuple[str, str, str], np.memmap] = {}
        self.schema: dict[str, dict[str, Any]] = {}
        self.data_hashes: dict[tuple[str, str, str], Any] = {}
        self.paths: dict[tuple[str, str, str], Path] = {}
        self.pending_bytes = 0

    def write(self, source: str, role: str, row: int, count: int, fields: Mapping[str, Any]) -> None:
        if not source or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in source):
            raise TransitionError("unsafe source identifier for indexed cache")
        source_schema = self.schema.setdefault(source, {})
        for field, raw in fields.items():
            value = np.asarray(raw)
            if value.dtype.hasobject:
                raise TransitionError(f"object dtype is forbidden in cache field {field}")
            schema = {"shape": list(value.shape), "dtype": value.dtype.str}
            if field in source_schema and any(source_schema[field].get(k) != v for k, v in schema.items()):
                raise TransitionError(f"schema changed for {source}/{field}")
            source_schema.setdefault(field, {}).update(schema)
            key = (source, role, field)
            if key not in self.maps:
                dest = self.root / source / role
                dest.mkdir(parents=True, exist_ok=True)
                filename = hashlib.sha256(field.encode("utf-8")).hexdigest()[:16] + ".npy"
                allocation = int(count) * int(value.nbytes) + 256
                if self.before_allocate:
                    self.before_allocate(allocation)
                self.maps[key] = np.lib.format.open_memmap(dest / filename, mode="w+",
                                                           dtype=value.dtype, shape=(count,) + value.shape)
                self.paths[key] = dest / filename
                self.data_hashes[key] = hashlib.sha256()
                source_schema[field][role + "_file"] = (dest / filename).relative_to(self.root.parent).as_posix()
            target = self.maps[key]
            if target.shape[1:] != value.shape or target.dtype != value.dtype or not 0 <= row < target.shape[0]:
                raise TransitionError(f"indexed write does not fit schema: {source}/{field}/{row}")
            target[row] = value
            self.data_hashes[key].update(value.tobytes(order="C"))
            self.pending_bytes += int(value.nbytes)

    def flush(self, *, close: bool = False) -> None:
        for array in self.maps.values():
            array.flush()
        if close:
            self.maps.clear()

    def verify_readback(self, guard: Callable[[int], None] | None = None) -> None:
        self.flush()
        for key, path in self.paths.items():
            if guard:
                guard(0)
            actual = hashlib.sha256()
            view = None
            try:
                view = np.load(path, mmap_mode="r", allow_pickle=False)
                if guard:
                    guard(0)
                for first in range(0, view.shape[0], 16):
                    if guard:
                        guard(0)
                    for row in view[first:first + 16]:
                        actual.update(np.asarray(row).tobytes(order="C"))
            finally:
                if view is not None:
                    mmap = getattr(view, "_mmap", None)
                    if mmap is not None:
                        mmap.close()
            if actual.digest() != self.data_hashes[key].digest():
                raise TransitionError(f"bounded cache readback mismatch for {key[0]}/{key[1]}/{key[2]}")

    def validate_layout(self, source_counts: Mapping[str, int]) -> None:
        for source, count in source_counts.items():
            fields = self.schema.get(source)
            if not fields:
                raise TransitionError(f"cache window schema is missing source {source}")
            for field, definition in fields.items():
                for role, expected_count in (("pre", int(count)), ("final_post", 1)):
                    relative = definition.get(role + "_file")
                    if not relative:
                        raise TransitionError(f"cache field coverage is missing {source}/{role}/{field}")
                    path = self.root.parent / relative
                    view = np.load(path, mmap_mode="r", allow_pickle=False)
                    try:
                        if view.shape != (expected_count, *definition["shape"]) or view.dtype.str != definition["dtype"]:
                            raise TransitionError(f"cache window shape/dtype mismatch for {source}/{role}/{field}")
                    finally:
                        mmap = getattr(view, "_mmap", None)
                        if mmap is not None:
                            mmap.close()


class _FrameIndexedWindowStore:
    """Lossless source-local native-frame arrays plus explicit window references."""
    INVALID_REF = np.iinfo(np.int64).min

    def __init__(self, root: Path, *, offsets: Sequence[int], temporal_fields: Sequence[str] | None = None,
                 before_allocate: Callable[[int], None] | None = None):
        self.root = Path(root)
        self.offsets = tuple(map(int, offsets))
        self.temporal_fields = frozenset(map(str, temporal_fields)) if temporal_fields is not None else None
        self.before_allocate = before_allocate
        self.schema: dict[str, dict[str, Any]] = {}
        self.maps: dict[tuple[str, str], np.memmap] = {}
        self.paths: dict[str, Path] = {}
        self.expected: dict[tuple[str, str, int], str] = {}
        self.source_meta: dict[str, dict[str, Any]] = {}
    @staticmethod
    def _safe(source: str) -> None:
        if not source or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in source):
            raise TransitionError("unsafe source identifier for indexed frame cache")

    def _array(self, relative: str, *, dtype: Any, shape: Sequence[int], fill: Any | None = None) -> np.memmap:
        path = self.root.parent / relative
        allocation = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize + 256
        if self.before_allocate:
            self.before_allocate(allocation)
        path.parent.mkdir(parents=True, exist_ok=True)
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
        if fill is not None:
            array[...] = fill
        self.paths[relative] = path
        return array

    def configure(self, source: str, rows: int, *, reset_frame: int, context_frames: int,
                  source_id: int = -1) -> None:
        self._safe(source)
        if source in self.source_meta:
            raise TransitionError(f"duplicate frame-indexed source identity: {source}")
        first_local = min(self.offsets)
        last_local = int(context_frames) + int(rows)
        capacity = last_local - first_local + 1
        if capacity <= 0:
            raise TransitionError(f"invalid source frame capacity for {source}")
        self.source_meta[source] = {"rows": int(rows), "source_id": int(source_id),
            "reset_frame": int(reset_frame),
            "context_frames": int(context_frames), "min_local_frame": first_local,
            "max_local_frame": last_local, "frame_capacity": capacity,
            "frame_valid_file": f"windows/{source}/frame_valid.npy",
            "frame_valid": None, "frame_fields": [], "padding": {},
            "roles": {}, "field_set": None}

    def _digest(self, fields: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        for name in sorted(fields):
            value = np.asarray(fields[name])
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(value.tobytes(order="C"))
        return digest.hexdigest()

    def _init_source_schema(self, source: str, role: str, count: int,
                            fields: Mapping[str, Any]) -> dict[str, Any]:
        meta = self.source_meta[source]
        schema = self.schema.setdefault(source, {})
        field_set = frozenset(map(str, fields))
        if meta["field_set"] is None:
            meta["field_set"] = field_set
        elif meta["field_set"] != field_set:
            raise TransitionError(f"model input field set changed inside source {source}")
        roles = meta["roles"]
        role_schema = roles.setdefault(role, {})
        if role not in ("pre", "final_post"):
            raise TransitionError(f"unsupported indexed window role: {role}")
        valid_window = np.asarray(fields.get("history_valid"))
        if valid_window.shape != (len(self.offsets),) or valid_window.dtype.hasobject:
            raise TransitionError(f"{source}/{role}: history_valid shape/dtype is unsupported")
        if role not in meta.get("role_refs", {}):
            base = f"windows/{source}/refs_{role}.npy"
            refs = self._array(base, dtype=np.int64, shape=(count, len(self.offsets)), fill=self.INVALID_REF)
            valid_path = f"windows/{source}/history_valid_{role}.npy"
            valid = self._array(valid_path, dtype=valid_window.dtype,
                                shape=(count, len(self.offsets)), fill=0)
            meta.setdefault("role_refs", {})[role] = refs
            meta["role_valid"] = meta.setdefault("role_valid", {})
            meta["role_valid"][role] = valid
            self.paths[base] = self.root.parent / base
            self.paths[valid_path] = self.root.parent / valid_path
            meta.setdefault("role_paths", {})[role] = {"refs_file": base,
                "history_valid_file": valid_path}
        temporal_fields = []
        for field, raw in fields.items():
            value = np.asarray(raw)
            if value.dtype.hasobject:
                raise TransitionError(f"object dtype is forbidden in cache field {field}")
            temporal = field != "history_valid" and (
                field in self.temporal_fields if self.temporal_fields is not None
                else value.ndim >= 1 and value.shape[0] == len(self.offsets))
            if temporal and (value.ndim < 1 or value.shape[0] != len(self.offsets)):
                raise TransitionError(f"{source}/{field}: preflight-declared temporal axis is absent")
            if field == "history_valid":
                schema[field] = {"shape": list(value.shape), "dtype": value.dtype.str,
                                 "temporal_axis": False, "role_files": {}}
                role_schema[field] = {"kind": "history_valid", "path": f"windows/{source}/history_valid_{role}.npy"}
                temporal_fields.append(field)
            elif temporal:
                if value.shape[0] != len(self.offsets):
                    raise TransitionError(f"{source}/{field}: temporal axis differs from verified offsets")
                feature_shape = tuple(value.shape[1:])
                field_schema = schema.setdefault(field, {"shape": list(feature_shape),
                    "dtype": value.dtype.str, "temporal_axis": True, "frames_file":
                        f"windows/{source}/frames/{hashlib.sha256(field.encode('utf-8')).hexdigest()[:16]}.npy",
                    "padding_file": f"windows/{source}/padding/{hashlib.sha256(field.encode('utf-8')).hexdigest()[:16]}.npy"})
                if field_schema["shape"] != list(feature_shape) or field_schema["dtype"] != value.dtype.str:
                    raise TransitionError(f"schema changed for {source}/{field}")
                role_schema[field] = {"kind": "frame_refs", "frames_file": field_schema["frames_file"]}
                temporal_fields.append(field)
                if field not in meta["frame_fields"]:
                    meta["frame_fields"].append(field)
                    path = field_schema["frames_file"]
                    self.maps[(source, field)] = self._array(path, dtype=value.dtype,
                        shape=(meta["frame_capacity"], *feature_shape))
                    if "frame_valid" not in meta or meta["frame_valid"] is None:
                        meta["frame_valid"] = self._array(f"windows/{source}/frame_valid.npy",
                            dtype=np.uint8, shape=(meta["frame_capacity"],), fill=0)
                if any(not bool(x) for x in valid_window) and field not in meta["padding"]:
                    padding_path = field_schema["padding_file"]
                    padding = self._array(padding_path, dtype=value.dtype, shape=feature_shape)
                    invalid_slots = [i for i, ok in enumerate(valid_window) if not bool(ok)]
                    padding[...] = value[invalid_slots[0]]
                    meta["padding"][field] = padding
                    field_schema["padding_present"] = True
                elif any(not bool(x) for x in valid_window):
                    padding = meta["padding"][field]
                    for i, ok in enumerate(valid_window):
                        if not bool(ok) and not np.array_equal(value[i], padding):
                            raise TransitionError(f"invalid prehistory padding changed for {source}/{field}")
            else:
                field_schema = schema.setdefault(field, {"shape": list(value.shape), "dtype": value.dtype.str,
                    "temporal_axis": False, "role_files": {}})
                if field_schema["shape"] != list(value.shape) or field_schema["dtype"] != value.dtype.str:
                    raise TransitionError(f"non-temporal schema changed for {source}/{field}")
                relative = f"windows/{source}/{role}/non_temporal_{hashlib.sha256(field.encode('utf-8')).hexdigest()[:16]}.npy"
                field_schema["role_files"][role] = relative
                role_schema[field] = {"kind": "non_temporal", "path": relative}
                key = (source, f"non_temporal:{role}:{field}")
                if key not in self.maps:
                    self.maps[key] = self._array(relative, dtype=value.dtype,
                        shape=(count, *value.shape))
            temporal_fields.append(field)
        return role_schema

    def write(self, source: str, role: str, row: int, count: int,
              fields: Mapping[str, Any], *, current_frame: int) -> int:
        meta = self.source_meta[source]
        role_schema = self._init_source_schema(source, role, count, fields)
        valid = np.asarray(fields["history_valid"])
        refs = meta["role_refs"][role]
        row_refs = np.full(len(self.offsets), self.INVALID_REF, dtype=np.int64)
        meta["role_valid"][role][row] = valid
        traffic_bytes = int(row_refs.nbytes + valid.nbytes)
        reset_frame = int(meta["reset_frame"])
        local_current = int(current_frame) - reset_frame
        for slot, offset in enumerate(self.offsets):
            if not bool(valid[slot]):
                continue
            local_ref = local_current + int(offset)
            index = local_ref - int(meta["min_local_frame"])
            if index < 0 or index >= int(meta["frame_capacity"]):
                raise TransitionError(f"{source}: valid history reference is outside source frame table")
            if meta["frame_valid"][index]:
                for field in meta["frame_fields"]:
                    existing = self.maps[(source, field)][index]
                    actual = np.asarray(fields[field])[slot]
                    if existing.dtype != actual.dtype or not np.array_equal(existing, actual, equal_nan=False):
                        raise TransitionError(f"shared native-frame value conflict for {source}@{local_ref}/{field}")
            else:
                for field in meta["frame_fields"]:
                    frame_value = np.asarray(fields[field])[slot]
                    self.maps[(source, field)][index] = frame_value
                    traffic_bytes += int(frame_value.nbytes)
                meta["frame_valid"][index] = 1
                traffic_bytes += 1
            row_refs[slot] = local_ref
        refs[row] = row_refs
        for field, definition in self.schema[source].items():
            if field == "history_valid" or definition["temporal_axis"]:
                continue
            self.maps[(source, f"non_temporal:{role}:{field}")][row] = np.asarray(fields[field])
            traffic_bytes += int(np.asarray(fields[field]).nbytes)
        self.expected[(source, role, int(row))] = self._digest(fields)
        return traffic_bytes

    def prepare(self, source: str, rows: int, fields: Mapping[str, Any]) -> None:
        """Reserve/create both lazy role indexes and all first-source arrays pre-dispatch."""
        self._init_source_schema(source, "pre", rows, fields)
        self._init_source_schema(source, "final_post", 1, fields)

    def reconstruct(self, source: str, role: str, row: int) -> dict[str, np.ndarray]:
        meta = self.source_meta[source]
        refs = meta["role_refs"][role]
        valid = np.asarray(meta["role_valid"][role][row]).copy()
        result: dict[str, np.ndarray] = {}
        for field, definition in self.schema[source].items():
            if field == "history_valid":
                result[field] = valid
            elif definition["temporal_axis"]:
                frames = self.maps[(source, field)]
                slots = []
                for slot, ref in enumerate(refs[row]):
                    if not bool(valid[slot]):
                        slots.append(np.asarray(meta["padding"][field]).copy())
                        continue
                    index = int(ref) - int(meta["min_local_frame"])
                    if int(ref) == self.INVALID_REF or index < 0 or index >= int(meta["frame_capacity"]):
                        raise TransitionError(f"missing or invalid frame reference {source}/{role}/{row}/{slot}")
                    if not meta["frame_valid"][index]:
                        raise TransitionError(f"evicted or missing frame generation {source}@{int(ref)}")
                    slots.append(np.asarray(frames[index]).copy())
                result[field] = np.stack(slots)
            else:
                result[field] = np.asarray(self.maps[(source, f"non_temporal:{role}:{field}")][row]).copy()
        return result

    def flush(self, *, close: bool = False) -> None:
        for array in self.maps.values():
            array.flush()
        for meta in self.source_meta.values():
            for roles in (meta.get("role_refs", {}), meta.get("role_valid", {})):
                for array in roles.values():
                    array.flush()
            if meta.get("frame_valid") is not None:
                meta["frame_valid"].flush()
        if close:
            arrays = list(self.maps.values())
            for meta in self.source_meta.values():
                arrays.extend(meta.get("role_refs", {}).values())
                arrays.extend(meta.get("role_valid", {}).values())
                if meta.get("frame_valid") is not None:
                    arrays.append(meta["frame_valid"])
                arrays.extend(meta.get("padding", {}).values())
            for array in arrays:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            self.maps.clear()

    def verify_readback(self, guard: Callable[[int], None] | None = None) -> None:
        self.flush()
        for (source, role, row), expected in self.expected.items():
            if guard:
                guard(0)
            actual = self.reconstruct(source, role, row)
            if self._digest(actual) != expected:
                raise TransitionError(f"frame-indexed independent window readback mismatch {source}/{role}/{row}")

    def validate_layout(self, source_counts: Mapping[str, int]) -> None:
        for source, count in source_counts.items():
            meta = self.source_meta.get(source)
            if not meta or meta["rows"] != int(count):
                raise TransitionError(f"frame-indexed source count mismatch for {source}")
            for role, rows in (("pre", int(count)), ("final_post", 1)):
                refs = meta.get("role_refs", {}).get(role)
                valid = meta.get("role_valid", {}).get(role)
                if refs is None or valid is None or refs.shape != (rows, 7):
                    raise TransitionError(f"frame-indexed refs/validity coverage missing for {source}/{role}")
                for row in range(rows):
                    self.reconstruct(source, role, row)

    def index_manifest(self) -> dict[str, Any]:
        return {source: {"rows": meta["rows"], "source_id": meta["source_id"],
            "reset_frame": meta["reset_frame"], "context_frames": meta["context_frames"],
            "min_local_frame": meta["min_local_frame"], "max_local_frame": meta["max_local_frame"],
            "frame_capacity": meta["frame_capacity"], "frame_valid_file": meta["frame_valid_file"],
            "roles": meta.get("role_paths", {})} for source, meta in self.source_meta.items()}


def _validate_measurements(result: Mapping[str, Any], location: str) -> None:
    try:
        reward = float(result.get("reward", math.nan))
        reward_ok = math.isfinite(reward)
    except (TypeError, ValueError):
        reward_ok = False
    if not reward_ok:
        raise TransitionError(f"{location}: reward is not finite")
    components, availability = result.get("components"), result.get("availability")
    if not isinstance(components, Mapping) or not isinstance(availability, Mapping):
        raise TransitionError(f"{location}: reward components/availability must be mappings")
    if set(components) != set(availability):
        raise TransitionError(f"{location}: component/availability keys differ")
    for name, value in components.items():
        try:
            valid_value = math.isfinite(float(value))
        except (TypeError, ValueError):
            valid_value = False
        if not valid_value or not isinstance(availability[name], (bool, np.bool_)):
            raise TransitionError(f"{location}: invalid component or availability for {name}")


class _FrameIndexedReader:
    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = Path(root)
        self._manifest_sources = manifest.get("sources", [])
        self.schema = manifest.get("window_schema", {})
        self.index = manifest.get("frame_index", {})
        self.maps: dict[tuple[str, str], np.ndarray] = {}
        self._load_and_validate()

    def _load(self, source: str, key: str, relative: str) -> np.ndarray:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise TransitionError(f"unsafe frame-index path: {relative}")
        path = self.root / relative
        if not path.is_file():
            raise TransitionError(f"frame-index file is missing: {relative}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise TransitionError(f"invalid frame-index NPY {relative}: {exc}") from exc
        self.maps[(source, key)] = array
        return array

    def _load_and_validate(self) -> None:
        if set(self.schema) != set(self.index):
            raise TransitionError("frame-index source set differs from window schema")
        source_summaries = {item.get("name"): item for item in self._manifest_sources}
        for source, source_meta in self.index.items():
            fields = self.schema[source]
            summary = source_summaries.get(source, {})
            for key in ("rows", "source_id", "reset_frame", "context_frames"):
                if key in summary and int(source_meta.get(key, -1)) != int(summary[key]):
                    raise TransitionError(f"frame-index source identity mismatch for {source}/{key}")
            capacity = int(source_meta.get("frame_capacity", -1))
            min_local, max_local = int(source_meta.get("min_local_frame", 0)), int(source_meta.get("max_local_frame", -1))
            if capacity != max_local - min_local + 1:
                raise TransitionError(f"invalid frame-index capacity for {source}")
            valid = self._load(source, "frame_valid", source_meta["frame_valid_file"])
            if valid.shape != (capacity,) or valid.dtype != np.dtype(np.uint8):
                raise TransitionError(f"frame-valid table shape/dtype mismatch for {source}")
            if np.any((valid != 0) & (valid != 1)):
                raise TransitionError(f"frame-valid table has unsupported values for {source}")
            source_role_specs = source_meta.get("roles", {})
            for field, definition in fields.items():
                if definition.get("temporal_axis"):
                    frames = self._load(source, f"frames:{field}", definition["frames_file"])
                    if frames.shape != (capacity, *definition["shape"]) or frames.dtype.str != definition["dtype"]:
                        raise TransitionError(f"encoded frame shape/dtype mismatch for {source}/{field}")
                    pad_path = definition.get("padding_file")
                    if pad_path and (self.root / pad_path).is_file():
                        pad = self._load(source, f"padding:{field}", pad_path)
                        if pad.shape != tuple(definition["shape"]) or pad.dtype.str != definition["dtype"]:
                            raise TransitionError(f"padding shape/dtype mismatch for {source}/{field}")
                for role, role_spec in source_role_specs.items():
                    if field == "history_valid":
                        path = role_spec.get("history_valid_file")
                        array = self._load(source, f"valid:{role}", path)
                        rows = int(source_meta["rows"]) if role == "pre" else 1
                        if array.shape != (rows, 7) or array.dtype.str != definition["dtype"]:
                            raise TransitionError(f"history validity shape/dtype mismatch for {source}/{role}")
                    elif not definition.get("temporal_axis"):
                        relative = definition.get("role_files", {}).get(role)
                        if not relative:
                            continue
                        array = self._load(source, f"non_temporal:{role}:{field}", relative)
                        rows = int(source_meta["rows"]) if role == "pre" else 1
                        if array.shape != (rows, *definition["shape"]) or array.dtype.str != definition["dtype"]:
                            raise TransitionError(f"non-temporal shape/dtype mismatch for {source}/{role}/{field}")
            for role, role_spec in source_role_specs.items():
                rows = int(source_meta["rows"]) if role == "pre" else 1
                refs = self._load(source, f"refs:{role}", role_spec["refs_file"])
                history = self.maps[(source, f"valid:{role}")]
                if refs.shape != (rows, 7) or refs.dtype != np.dtype(np.int64):
                    raise TransitionError(f"frame reference table shape/dtype mismatch for {source}/{role}")
                for row in range(rows):
                    for slot in range(7):
                        ref, is_valid = int(refs[row, slot]), bool(history[row, slot])
                        if not is_valid:
                            if ref != _FrameIndexedWindowStore.INVALID_REF:
                                raise TransitionError(f"invalid history slot has a fabricated reference {source}/{role}/{row}/{slot}")
                            continue
                        offset = ref - min_local
                        if ref < min_local or ref > max_local or not valid[offset]:
                            raise TransitionError(f"missing, evicted or stale frame reference {source}/{role}/{row}/{slot}")

    def window(self, source: str, role: str, row: int) -> dict[str, np.ndarray]:
        if source not in self.index or role not in self.index[source].get("roles", {}):
            raise TransitionError(f"unknown frame-indexed window {source}/{role}")
        meta, role_spec = self.index[source], self.index[source]["roles"][role]
        rows = int(meta["rows"]) if role == "pre" else 1
        if not 0 <= int(row) < rows:
            raise TransitionError(f"window row out of range {source}/{role}/{row}")
        valid = np.asarray(self.maps[(source, f"valid:{role}")][row]).copy()
        refs = self.maps[(source, f"refs:{role}")][row]
        output: dict[str, np.ndarray] = {}
        for field, definition in self.schema[source].items():
            if field == "history_valid":
                output[field] = valid
            elif definition.get("temporal_axis"):
                frames = self.maps[(source, f"frames:{field}")]
                padding = self.maps.get((source, f"padding:{field}"))
                slots = []
                for slot in range(7):
                    if not bool(valid[slot]):
                        if padding is None:
                            raise TransitionError(f"padding payload is missing for invalid slot {source}/{field}")
                        slots.append(np.asarray(padding).copy())
                    else:
                        ref = int(refs[slot])
                        idx = ref - int(meta["min_local_frame"])
                        if ref == _FrameIndexedWindowStore.INVALID_REF or idx < 0 or idx >= int(meta["frame_capacity"]):
                            raise TransitionError(f"missing frame reference {source}/{role}/{row}/{slot}")
                        slots.append(np.asarray(frames[idx]).copy())
                output[field] = np.stack(slots)
            else:
                key = (source, f"non_temporal:{role}:{field}")
                array = self.maps.get(key)
                if array is None:
                    raise TransitionError(f"non-temporal field reference missing for {source}/{role}/{field}")
                output[field] = np.asarray(array[row]).copy()
        return output

    def validate_identity(self, source: str, role: str, row: int, current_frame: int,
                          offsets: Sequence[int]) -> None:
        meta = self.index[source]
        refs = self.maps[(source, f"refs:{role}")][int(row)]
        valid = self.maps[(source, f"valid:{role}")][int(row)]
        current_local = int(current_frame) - int(meta["reset_frame"])
        for slot, offset in enumerate(offsets):
            if bool(valid[slot]):
                if int(refs[slot]) != current_local + int(offset):
                    raise TransitionError(f"frame reference identity/offset mismatch {source}/{role}/{row}/{slot}")
            elif int(refs[slot]) != _FrameIndexedWindowStore.INVALID_REF:
                raise TransitionError(f"invalid slot has a frame identity {source}/{role}/{row}/{slot}")

    def close(self) -> None:
        for array in self.maps.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.maps.clear()


class TransitionCache:
    """Read-only integrity-checked cache reader; partial output is rejected."""
    def __init__(self, root: Path, *, unpublished: bool = False):
        self.root = Path(root)
        if (not unpublished and not (self.root / "READY").is_file()) or not (self.root / "manifest.json").is_file():
            raise TransitionError("cache is partial or not finalized")
        if not unpublished:
            ready = json.loads((self.root / "READY").read_text(encoding="utf-8"))
            if ready.get("manifest_sha256") != sha256_file(self.root / "manifest.json"):
                raise TransitionError("manifest hash mismatch")
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.format = self.manifest.get("format")
        if self.format not in (FORMAT, LEGACY_FORMAT) or self.manifest.get("status") != "ready":
            raise TransitionError("unsupported or unready cache manifest")
        for relative, digest in self.manifest.get("files", {}).items():
            target = self.root / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise TransitionError(f"unsafe cache file reference: {relative}")
            if not target.is_file() or sha256_file(target) != digest:
                raise TransitionError(f"cache file hash mismatch: {relative}")
        if self.format == FORMAT:
            identity_ref = self.manifest.get("identity_ref")
            if (not isinstance(identity_ref, Mapping) or identity_ref.get("path") != "identity.json"
                    or identity_ref.get("sha256") != sha256_file(self.root / "identity.json")):
                raise TransitionError("immutable runtime identity receipt is missing or hash-mismatched")
            identity = json.loads((self.root / "identity.json").read_text(encoding="utf-8"))
            if identity.get("attempt_id") != self.manifest.get("attempt_id"):
                raise TransitionError("runtime identity attempt differs from cache manifest")
            outcome_spec = self.manifest.get("outcome_receipts")
            context_spec = self.manifest.get("context_receipts", {})
            if not isinstance(outcome_spec, Mapping) or outcome_spec.get("path") != "outcomes.jsonl":
                raise TransitionError("frame-indexed cache lacks its outcome evidence journal")
            outcome_count = int(self.manifest.get("rows", -1)) + int(context_spec.get("count", -1))
            actual_outcome_bytes = _validate_outcomes(self.root / "outcomes.jsonl",
                attempt_id=str(self.manifest.get("attempt_id", "")), expected_count=outcome_count)
            if (int(outcome_spec.get("count", -1)) != outcome_count
                    or actual_outcome_bytes != (self.root / "outcomes.jsonl").stat().st_size):
                raise TransitionError("outcome evidence count/size differs from the ready manifest")
            with (self.root / "outcomes.jsonl").open("r", encoding="utf-8") as outcomes:
                for line in outcomes:
                    payload = json.loads(line)["payload"]
                    if payload.get("parity_validated") is not False or payload.get("accepted_demonstration_row") is not False:
                        raise TransitionError("failure-evidence outcome was marked as an accepted demonstration row")
        self._frame_reader = (_FrameIndexedReader(self.root, self.manifest)
                              if self.format == FORMAT else None)
        source_counts = {item["name"]: int(item["rows"]) for item in self.manifest.get("sources", [])}
        source_meta = {item["name"]: item for item in self.manifest.get("sources", [])}
        if self.format == LEGACY_FORMAT:
            self._validate_legacy_layout(source_counts)
        offsets = tuple(map(int, self.manifest.get("history_offsets", ())))
        if self.format == FORMAT and len(offsets) != 7:
            raise TransitionError("frame-indexed manifest lacks the seven verified history offsets")
        if self.format == FORMAT:
            context_spec = self.manifest.get("context_receipts")
            if not isinstance(context_spec, Mapping) or context_spec.get("path") != "context_receipts.jsonl":
                raise TransitionError("cache context receipt index is missing or unsafe")
            context_records = read_records(self.root / "context_receipts.jsonl",
                                           attempt_id=str(self.manifest.get("attempt_id", "")))
            if len(context_records) != int(context_spec.get("count", -1)):
                raise TransitionError("context receipt count differs from manifest")
            for source in self.manifest.get("sources", []):
                ref = source.get("context_ref", {})
                first, last, expected = (int(ref.get("first_sequence", 0)),
                                         int(ref.get("last_sequence", -1)),
                                         int(ref.get("count", -1)))
                selected = context_records[first - 1:last] if expected else []
                if (len(selected) != expected or expected != int(source.get("context_frames", -2))
                        or any(item.get("payload", {}).get("source") != source.get("name") for item in selected)
                        or [int(item["sequence"]) for item in selected] != list(range(first, last + 1))
                        or hashlib.sha256("".join(str(item["sha256"]) for item in selected).encode("ascii")).hexdigest()
                        != ref.get("sha256")):
                    raise TransitionError(f"context receipt reference/hash mismatch for {source.get('name')}")
        records = self.root / "transitions.jsonl"
        count = 0
        with records.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                owner = record.get("trajectory")
                ref = record.get("next_ref", {})
                if owner not in source_counts:
                    raise TransitionError("transition references unknown trajectory")
                local = int(record.get("local_row", -1))
                src = source_meta[owner]
                if local < 0 or local >= source_counts[owner]:
                    raise TransitionError("transition local row is outside its source partition")
                if (record.get("source_id") is not None and "source_id" in src
                        and int(record["source_id"]) != int(src["source_id"])):
                    raise TransitionError("transition source ID differs from source-local frame identity")
                if "source_kind" in src and record.get("source_kind") != src.get("source_kind"):
                    raise TransitionError("transition source kind differs from manifest provenance")
                if "start_frame" in src and int(record.get("frame", -1)) != int(src["start_frame"]) + local:
                    raise TransitionError("transition frame order differs from source manifest")
                if self._frame_reader is not None:
                    self._frame_reader.validate_identity(owner, "pre", local, int(record["frame"]), offsets)
                if bool(record.get("bootstrap_valid")) != bool(record.get("bootstrap")):
                    raise TransitionError("environment bootstrap receipt differs from cache boundary metadata")
                if ref.get("kind") == "pre":
                    if ref.get("trajectory", owner) != owner or not 0 <= int(ref.get("row", -1)) < source_counts[owner]:
                        raise TransitionError("unsafe or cross-trajectory following reference")
                    if int(record.get("local_row", -1)) + 1 != int(ref["row"]):
                        raise TransitionError("interior following reference is not the next local row")
                elif ref.get("kind") == "final_post":
                    if ref.get("trajectory") != owner or int(ref.get("row", -1)) != int(record.get("local_row", -2)):
                        raise TransitionError("final post reference does not belong to its final row")
                    if record.get("natural_terminal") and record.get("bootstrap"):
                        raise TransitionError("natural terminal final row cannot bootstrap")
                    if record.get("time_limit") and not record.get("bootstrap"):
                        raise TransitionError("time-limit final row must preserve bootstrap eligibility")
                    if not record.get("natural_terminal") and not record.get("time_limit"):
                        raise TransitionError("final post row lacks a valid boundary")
                    if self._frame_reader is not None:
                        self._frame_reader.validate_identity(owner, "final_post", 0,
                            int(record["frame"]) + 1, offsets)
                else:
                    raise TransitionError("unknown following reference kind")
                count += 1
        if count != int(self.manifest.get("rows", -1)):
            raise TransitionError("manifest transition row count mismatch")
        self._records = (self.root / "transitions.jsonl").open("r", encoding="utf-8")

    def _validate_legacy_layout(self, source_counts: Mapping[str, int]) -> None:
        schema = self.manifest.get("window_schema", {})
        if set(schema) != set(source_counts):
            raise TransitionError("legacy window schema source set is incomplete")
        for source, count in source_counts.items():
            if not schema[source]:
                raise TransitionError(f"legacy window schema is empty for {source}")
            for field, definition in schema[source].items():
                for role, rows in (("pre", int(count)), ("final_post", 1)):
                    relative = definition.get(role + "_file")
                    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise TransitionError(f"legacy window field is missing/unsafe: {source}/{role}/{field}")
                    path = self.root / relative
                    if not path.is_file():
                        raise TransitionError(f"legacy window file is missing: {relative}")
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    try:
                        expected_shape = (rows, *definition["shape"])
                        if array.shape != expected_shape or array.dtype.str != definition["dtype"]:
                            raise TransitionError(f"legacy window shape/dtype mismatch for {source}/{role}/{field}")
                    finally:
                        mmap = getattr(array, "_mmap", None)
                        if mmap is not None:
                            mmap.close()

    def __iter__(self):
        for line in self._records:
            yield json.loads(line)

    def close(self):
        self._records.close()
        if self._frame_reader is not None:
            self._frame_reader.close()

    def window(self, source: str, role: str, row: int) -> dict[str, np.ndarray]:
        if self._frame_reader is not None:
            return self._frame_reader.window(source, role, row)
        if source not in self.manifest.get("window_schema", {}):
            raise TransitionError(f"unknown legacy cache source {source}")
        result = {}
        source_schema = self.manifest["window_schema"][source]
        for field, definition in source_schema.items():
            relative = definition.get(role + "_file")
            if not relative:
                raise TransitionError(f"legacy cache field missing {source}/{role}/{field}")
            path = self.root / relative
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            try:
                result[field] = np.asarray(array[int(row)]).copy()
            finally:
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def inspect_npz(path: Path) -> dict[str, Any]:
    """Bounded metadata and source partition preflight; no full dataset arrays."""
    import zipfile
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version in ((2, 0), (3, 0)):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise TransitionError(f"unsupported NPY header version {version}: {member}")
                headers[member[:-4]] = {"shape": list(shape), "dtype": np.dtype(dtype).str,
                                        "fortran_order": bool(fortran), "nbytes": int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize}
    return {"sha256": sha256_file(path), "headers": headers}
