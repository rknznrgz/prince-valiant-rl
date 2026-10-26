"""CPU source-replay resource sampling and owned-child wall supervision."""
from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .dqfd_transitions import TransitionError
from .dqfd_receipts import append_record, latest_record
from .dqfd_receipts import read_records

GIB = 1024 ** 3
RUNTIME_THRESHOLDS = {"available_ram_bytes": 4 * GIB, "commit_headroom_bytes": 4 * GIB,
                      "disk_reserve_bytes": GIB}
ML_LAUNCH_THRESHOLDS = {"available_ram_bytes": 8 * GIB, "commit_headroom_bytes": 12 * GIB,
                        "gpu_free_bytes": 6 * GIB}
ML_RUNTIME_THRESHOLDS = {"available_ram_bytes": 4 * GIB, "commit_headroom_bytes": 4 * GIB,
                         "gpu_free_bytes": 2 * GIB, "disk_reserve_bytes": GIB}
LOG_CAP_BYTES = 12 * 1024 * 1024
CHILD_LOG_CAP_BYTES = 4 * 1024 * 1024
IMMUTABLE_RECEIPT_CAP_BYTES = 64 * 1024


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t), ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t), ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_uint32),
        ("ProcessCount", ctypes.c_uint32), ("ThreadCount", ctypes.c_uint32)]


def resource_sample(stage: str, volume: Path, owned_pid: int | None = None, *,
                    wall_clock: Callable[[], float] = time.time,
                    monotonic_clock: Callable[[], float] = time.monotonic,
                    sample_id: str | None = None) -> dict[str, Any]:
    """Take one read-only Windows host/disk sample; denied metrics stay unavailable."""
    import datetime
    started_wall = float(wall_clock()); started_mono = float(monotonic_clock())
    row: dict[str, Any] = {"stage": stage, "sample_id": sample_id or uuid.uuid4().hex,
        "measurement_started_unix_time": started_wall,
        "measurement_started_at_utc": datetime.datetime.fromtimestamp(
            started_wall, datetime.timezone.utc).isoformat(),
        "host_read_started_unix_time": started_wall,
        "thresholds": dict(RUNTIME_THRESHOLDS), "available_ram_bytes": None,
        "physical_total_bytes": None, "committed_bytes": None, "commit_limit_bytes": None,
        "commit_headroom_bytes": None, "disk_free_bytes": None,
        "disk_target": Path(volume).resolve().drive, "owned_pid": owned_pid,
        "owned_process_tree": [], "sampling_failures": [], "unavailable_metrics": []}
    try:
        import psutil
        vm = psutil.virtual_memory()
        row["available_ram_bytes"] = int(vm.available)
        row["physical_total_bytes"] = int(vm.total)
        if owned_pid is not None:
            root = psutil.Process(int(owned_pid))
            owned = [root, *root.children(recursive=True)]
            row["owned_process_tree"] = [{"pid": p.pid, "name": p.name(), "status": p.status()}
                                          for p in owned if p.is_running()]
    except Exception as exc:
        row["sampling_failures"].append({"metric": "available_ram_or_process_tree", "error": repr(exc)[:300]})
    try:
        if os.name != "nt":
            raise OSError("Windows commit sampler unavailable on this platform")
        info = PERFORMANCE_INFORMATION(); info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise ctypes.WinError()
        row.update({"commit_total_pages": int(info.CommitTotal),
            "commit_limit_pages": int(info.CommitLimit), "system_page_size_bytes": int(info.PageSize),
            "committed_bytes": int(info.CommitTotal) * int(info.PageSize),
            "commit_limit_bytes": int(info.CommitLimit) * int(info.PageSize),
            "commit_headroom_bytes": (int(info.CommitLimit) - int(info.CommitTotal)) * int(info.PageSize)})
    except Exception as exc:
        row["sampling_failures"].append({"metric": "commit_headroom", "error": repr(exc)[:300]})
    try:
        row["disk_free_bytes"] = int(shutil.disk_usage(volume).free)
    except Exception as exc:
        row["sampling_failures"].append({"metric": "disk_free", "error": repr(exc)[:300]})
    for key in ("available_ram_bytes", "physical_total_bytes", "committed_bytes",
                "commit_limit_bytes", "commit_headroom_bytes", "disk_free_bytes"):
        if row[key] is None:
            row["unavailable_metrics"].append(key)
    completed_wall = float(wall_clock()); completed_mono = float(monotonic_clock())
    row.update({"host_read_completed_unix_time": completed_wall,
        "host_read_completed_at_utc": datetime.datetime.fromtimestamp(
            completed_wall, datetime.timezone.utc).isoformat(),
        "host_sampled_unix_time": completed_wall,
        "host_sampled_at_utc": datetime.datetime.fromtimestamp(
            completed_wall, datetime.timezone.utc).isoformat(),
        "sample_unix_time": completed_wall,
        "measured_at_utc": datetime.datetime.fromtimestamp(
            completed_wall, datetime.timezone.utc).isoformat(),
        "host_query_elapsed_seconds": max(0.0, completed_mono - started_mono),
        "host_clock_elapsed_consistent": completed_mono >= started_mono})
    return row


def gpu_free_memory_sample(*, wall_clock: Callable[[], float] = time.time,
                           monotonic_clock: Callable[[], float] = time.monotonic,
                           sample_id: str | None = None,
                           timeout_seconds: float = 10.0) -> dict[str, Any]:
    """Read free CUDA memory without creating a model or running an optimizer."""
    overall_wall = float(wall_clock()); overall_mono = float(monotonic_clock())
    result: dict[str, Any] = {"sample_id": sample_id or uuid.uuid4().hex,
        "gpu_probe_started_unix_time": overall_wall, "gpu_measurement_status": "failed",
        "gpu_free_bytes": None, "gpu_total_bytes": None, "gpu_measurement_error": None}
    if timeout_seconds <= 0:
        result["gpu_measurement_error"] = "GPU probe timeout must be positive"
        return result
    completed: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
    def probe() -> None:
        row = dict(result)
        try:
            import torch
            if not torch.cuda.is_available():
                row.update({"gpu_measurement_status": "missing",
                    "gpu_measurement_error": "CUDA device unavailable"})
            else:
                read_started_wall = float(wall_clock()); read_started_mono = float(monotonic_clock())
                free, total = torch.cuda.mem_get_info()
                read_completed_wall = float(wall_clock()); read_completed_mono = float(monotonic_clock())
                row.update({"gpu_free_bytes": int(free), "gpu_total_bytes": int(total),
                    "gpu_measurement_status": "healthy", "gpu_read_started_unix_time": read_started_wall,
                    "gpu_read_completed_unix_time": read_completed_wall,
                    "gpu_measured_unix_time": read_completed_wall,
                    "gpu_query_elapsed_seconds": max(0.0, read_completed_mono - read_started_mono),
                    "gpu_probe_elapsed_seconds": max(0.0, read_completed_mono - overall_mono),
                    "gpu_clock_elapsed_consistent": (read_completed_mono >= read_started_mono
                        and read_started_mono >= overall_mono)})
        except Exception as exc:
            row.update({"gpu_free_bytes": None, "gpu_measurement_status": "failed",
                "gpu_measurement_error": repr(exc)[:500]})
        try: completed.put_nowait(row)
        except queue.Full: pass
    worker = threading.Thread(target=probe, name="dqfd-gpu-resource-probe", daemon=True)
    worker.start(); worker.join(float(timeout_seconds))
    elapsed = max(0.0, float(monotonic_clock()) - overall_mono)
    if worker.is_alive() or elapsed > float(timeout_seconds):
        result.update({"gpu_measurement_status": "failed",
            "gpu_measurement_error": f"GPU probe timed out after {float(timeout_seconds):.3f} seconds",
            "gpu_probe_elapsed_seconds": elapsed, "gpu_clock_elapsed_consistent": True})
        return result
    try:
        return completed.get_nowait()
    except queue.Empty:
        result.update({"gpu_measurement_status": "failed",
            "gpu_measurement_error": f"GPU probe timed out after {float(timeout_seconds):.3f} seconds",
            "gpu_probe_elapsed_seconds": elapsed,
            "gpu_clock_elapsed_consistent": True})
        return result


def ml_resource_sample(stage: str, volume: Path, *, base_sampler=resource_sample,
                       gpu_sampler=gpu_free_memory_sample, owned_pid: int | None = None,
                       wall_clock: Callable[[], float] = time.time,
                       monotonic_clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Initialize/query GPU, then take host metrics for one coherent snapshot."""
    sample_id = uuid.uuid4().hex
    started = float(wall_clock()); started_mono = float(monotonic_clock())
    gpu = dict(gpu_sampler())
    gpu_id = gpu.get("sample_id")
    row = base_sampler(stage, volume, owned_pid)
    row.update(gpu)
    if gpu_id is not None:
        row["sample_id"] = sample_id
        row["gpu_sample_id"] = gpu_id
    row["measurement_started_unix_time"] = started
    import datetime
    row["measurement_started_at_utc"] = datetime.datetime.fromtimestamp(
        started, datetime.timezone.utc).isoformat()
    row["snapshot_elapsed_seconds"] = max(0.0, float(monotonic_clock()) - started_mono)
    row["sample_completed_unix_time"] = float(wall_clock())
    row["sample_completed_at_utc"] = __import__("datetime").datetime.fromtimestamp(
        row["sample_completed_unix_time"], __import__("datetime").timezone.utc).isoformat()
    row["sampled_at_unix"] = row["sample_completed_unix_time"]
    row["measurement_schema"] = "dqfd_ml_resource_snapshot_v2"
    return row


def validate_ml_resource_sample(row: dict[str, Any], *, launch: bool,
                                projected_remaining_write_bytes: int = 0,
                                wall_clock: Callable[[], float] = time.time) -> list[Any]:
    floors = ML_LAUNCH_THRESHOLDS if launch else ML_RUNTIME_THRESHOLDS
    failures: list[Any] = list(row.get("sampling_failures", []))
    validation_time = float(wall_clock())
    row["validation_unix_time"] = validation_time
    row["validation_at_utc"] = __import__("datetime").datetime.fromtimestamp(
        validation_time, __import__("datetime").timezone.utc).isoformat()
    for component, completed_key, alias_key in (
            ("gpu", "gpu_read_completed_unix_time", "gpu_measured_unix_time"),
            ("host", "host_read_completed_unix_time", "host_sampled_unix_time")):
        measured = row.get(completed_key)
        alias = row.get(alias_key)
        age = None if measured is None else validation_time - float(measured)
        consistent = (measured is not None and alias is not None
            and float(measured) == float(alias)
            and row.get("sample_id") is not None
            and row.get("measurement_schema") == "dqfd_ml_resource_snapshot_v2")
        started_key = "gpu_read_started_unix_time" if component == "gpu" else "host_read_started_unix_time"
        started_at = row.get(started_key)
        duration_key = "gpu_query_elapsed_seconds" if component == "gpu" else "host_query_elapsed_seconds"
        duration = row.get(duration_key)
        duration_valid = duration is not None and float(duration) >= 0.0
        ordered = (started_at is not None and measured is not None
            and float(started_at) <= float(measured)
            and (component != "gpu" or (row.get("gpu_probe_started_unix_time") is not None
                and float(row["gpu_probe_started_unix_time"]) <= float(started_at))))
        if component == "gpu":
            consistent = consistent and row.get("gpu_measurement_status") == "healthy"
        else:
            consistent = consistent and row.get("gpu_sample_id") is not None
        monotonic_ok = row.get("gpu_clock_elapsed_consistent", True) if component == "gpu" else row.get("host_clock_elapsed_consistent", True)
        if (not consistent or not ordered or not duration_valid or not monotonic_ok
                or age is None or age < 0.0 or age > 2.0):
            status = row.get("gpu_measurement_status", "missing") if component == "gpu" else "missing"
            if age is not None and (age < 0.0 or age > 2.0): status = "stale" if age > 2.0 else "clock_discontinuity"
            failures.append({"metric": f"{component}_measurement_freshness", "measured_unix_time": measured,
                "alias_unix_time": alias, "age_seconds": age, "maximum_age_seconds": 2.0,
                "measurement_status": status, "sample_id": row.get("sample_id"),
                "schema": row.get("measurement_schema"), "read_started_unix_time": started_at,
                "elapsed_query_seconds": duration})
    for key in ("available_ram_bytes", "commit_headroom_bytes", "gpu_free_bytes"):
        value = row.get(key)
        minimum = floors[key]
        if value is None or int(value) < minimum:
            failures.append({"metric": key, "measured": value, "minimum": minimum,
                "measurement_status": row.get("gpu_measurement_status") if key == "gpu_free_bytes" else None,
                "measurement_error": row.get("gpu_measurement_error") if key == "gpu_free_bytes" else None})
    disk = row.get("disk_free_bytes")
    needed = max(GIB, int(projected_remaining_write_bytes) + GIB)
    if disk is None or int(disk) < needed:
        failures.append({"metric": "disk_free_bytes", "measured": disk, "minimum": needed})
    return failures


def validate_resource_sample(row: dict[str, Any], *, launch: bool,
                             projected_remaining_write_bytes: int = 0) -> list[str]:
    floors = ({"available_ram_bytes": 8 * GIB, "commit_headroom_bytes": 12 * GIB}
              if launch else {"available_ram_bytes": 4 * GIB, "commit_headroom_bytes": 4 * GIB})
    failures = list(row.get("sampling_failures", []))
    for key, minimum in floors.items():
        value = row.get(key)
        if value is None or int(value) < minimum:
            failures.append({"metric": key, "measured": value, "minimum": minimum})
    disk = row.get("disk_free_bytes")
    reserve = GIB if not launch else int(row.get("artifact_cap_bytes", 4 * GIB)) + GIB
    needed = max(reserve, int(projected_remaining_write_bytes) + GIB)
    if disk is None or int(disk) < needed:
        failures.append({"metric": "disk_free_bytes", "measured": disk, "minimum": needed})
    return failures


def _append_bounded(path: Path, row: dict[str, Any], *, attempt_id: str, sequence: int) -> None:
    try:
        append_record(path, row, attempt_id=attempt_id, sequence=sequence, max_bytes=LOG_CAP_BYTES)
    except Exception as exc:
        raise TransitionError(f"resource journal append failed: {exc}") from exc


def _artifact_bytes(output: Path) -> int:
    """Measure retained child artifacts without charging supervisor diagnostics."""
    if not Path(output).exists(): return 0
    total = 0
    for path in Path(output).rglob("*"):
        if path.is_file(): total += int(path.stat().st_size)
    return total


def supervise_replay(command: list[str], *, cwd: Path, report_dir: Path, output: Path,
                     frame_cap: int, artifact_cap: int, wall_cap: float,
                     slice_cap: float | None = None,
                     sample_fn: Callable[[str, Path, int | None], dict[str, Any]] = resource_sample,
                     require_gpu: bool = False,
                     gpu_sampler: Callable[[], dict[str, Any]] = gpu_free_memory_sample,
                     clock: Callable[[], float] = time.monotonic,
                     wall_clock: Callable[[], float] = time.time,
                     monotonic_clock: Callable[[], float] = time.monotonic,
                     sleeper: Callable[[float], None] = time.sleep,
                     popen_factory: Callable[..., Any] = subprocess.Popen,
                     sample_period: float = 2.0,
                     started_at: float | None = None,
                     attempt_id: str | None = None,
                     child_receipt_path: Path | None = None,
                     child_attempt_id: str | None = None,
                     expected_child_exe: str | None = None,
                     artifact_size_fn: Callable[[Path], int] = _artifact_bytes) -> dict[str, Any]:
    """Run replay in one owned process and sample through its complete lifetime."""
    report_dir, output = Path(report_dir), Path(output)
    if output.exists():
        raise TransitionError(f"output root is occupied: {output}")
    if not 0 < sample_period <= 2.0:
        raise ValueError("resource watchdog sampling period must be at most two seconds")
    if slice_cap is not None and (not math.isfinite(float(slice_cap)) or float(slice_cap) <= 0):
        raise ValueError("supervised slice cap must be finite and positive")
    def elapsed_cap_reached(elapsed: float) -> bool:
        return elapsed >= wall_cap or (slice_cap is not None and elapsed >= slice_cap)
    log = report_dir / "runtime_resources.jsonl"
    report_dir.mkdir(parents=True, exist_ok=True)
    reserved = (log, report_dir / "reconstruction_resource_launch_receipt.json",
        report_dir / "progress.jsonl",
        report_dir / "replay.stdout.log",
        report_dir / "replay.stderr.log", report_dir / "supervisor_error.json",
        report_dir / "owned_process_stop_receipt.json", report_dir / "STOP_REQUESTED")
    occupied = [str(path) for path in reserved if path.exists()]
    if occupied:
        raise TransitionError(f"supervisor output is occupied; preserving existing paths: {occupied}")
    attempt_started = clock() if started_at is None else float(started_at)
    attempt_id = str(attempt_id or uuid.uuid4().hex)
    sample_sequence = 0
    def take_sample(stage: str, pid: int | None) -> dict[str, Any]:
        started_wall = float(wall_clock()); started_mono = float(monotonic_clock())
        gpu = dict(gpu_sampler()) if require_gpu else {}
        row = sample_fn(stage, output.parent, pid)
        row.update(gpu)
        if require_gpu:
            row["sample_id"] = uuid.uuid4().hex
            row["gpu_sample_id"] = gpu.get("sample_id")
            row["measurement_schema"] = "dqfd_ml_resource_snapshot_v2"
            row["measurement_started_unix_time"] = started_wall
            import datetime
            row["measurement_started_at_utc"] = datetime.datetime.fromtimestamp(
                started_wall, datetime.timezone.utc).isoformat()
            row["snapshot_elapsed_seconds"] = max(0.0, float(monotonic_clock()) - started_mono)
            row["sample_completed_unix_time"] = float(wall_clock())
            import datetime
            row["measurement_started_at_utc"] = datetime.datetime.fromtimestamp(
                started_wall, datetime.timezone.utc).isoformat()
            row["sample_completed_at_utc"] = datetime.datetime.fromtimestamp(
                row["sample_completed_unix_time"], datetime.timezone.utc).isoformat()
        return row
    def child_failure_link() -> dict[str, Any]:
        if child_receipt_path is None or child_attempt_id is None:
            return {"child_failure_receipt": "unknown"}
        try:
            records = read_records(Path(child_receipt_path), attempt_id=str(child_attempt_id),
                allow_truncated_tail=True)
            if not records:
                return {"child_failure_receipt": "not_written"}
            row = records[-1]
            payload = row.get("payload", {})
            return {"child_failure_receipt": "verified", "child_receipt_sequence": row["sequence"],
                "child_receipt_sha256": row["sha256"], "child_kind": payload.get("kind"),
                "child_counters": payload.get("counters"), "last_episode": payload.get("last_episode"),
                "child_error": payload.get("error")}
        except Exception as read_error:
            return {"child_failure_receipt": "unreadable", "child_receipt_error": repr(read_error)[:400]}
    def validate(row: dict[str, Any], *, launch: bool, projected: int) -> list[Any]:
        return (validate_ml_resource_sample(row, launch=launch,
            projected_remaining_write_bytes=projected, wall_clock=wall_clock) if require_gpu else
            validate_resource_sample(row, launch=launch,
            projected_remaining_write_bytes=projected))
    def executing_python_pid(root_pid: int) -> tuple[int, list[dict[str, Any]]]:
        """Resolve the venv launcher child by exact executable and retain creation identities."""
        if expected_child_exe is None:
            return int(root_pid), [{"pid": int(root_pid), "role": "supervised_process"}]
        import psutil
        try:
            root = psutil.Process(int(root_pid))
            candidates = [root, *root.children(recursive=True)]
            matches = []; identities = []
            for process in candidates:
                try:
                    record = {"pid": int(process.pid), "ppid": int(process.ppid()),
                        "create_time": float(process.create_time()), "exe": process.exe()}
                    identities.append(record)
                    if os.path.normcase(record["exe"]) == os.path.normcase(expected_child_exe):
                        matches.append(record)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            direct_matches = [row for row in matches if int(row["pid"]) == int(root_pid)
                or int(row["ppid"]) == int(root_pid)]
            if len(direct_matches) > 1:
                raise TransitionError("supervised tree has ambiguous direct Python identities")
            return (direct_matches[0]["pid"], identities) if direct_matches else (int(root_pid), identities)
        except psutil.NoSuchProcess:
            return int(root_pid), []
    launch = take_sample("launch_gate", None)
    launch["attempt_id"] = attempt_id
    launch["artifact_cap_bytes"] = int(artifact_cap)
    launch_failures = validate(launch, launch=True, projected=int(artifact_cap))
    launch["projected_remaining_write_bytes"] = int(artifact_cap)
    launch["runtime_gate_status"] = "failed" if launch_failures else "passed"
    sample_sequence += 1
    _append_bounded(log, launch, attempt_id=attempt_id, sequence=sample_sequence)
    if launch_failures:
        failure = TransitionError(f"launch resource gate failed: {launch_failures}")
        try:
            _atomic_receipt(report_dir / "supervisor_error.json", {"attempt_id": attempt_id,
                "status": "blocked_before_child", "trigger_sample": launch,
                "failures": launch_failures, "owned_pid": None})
        except Exception as receipt_exc:
            failure.add_note(f"launch error receipt also failed: {receipt_exc!r}")
        raise failure
    launch_path = report_dir / "reconstruction_resource_launch_receipt.json"
    _atomic_receipt(launch_path, {"attempt_id": attempt_id, "phase": "cpu_source_reconstruction", "status": "passed",
        "measured_at_utc": launch["measured_at_utc"], "available_ram_bytes": launch["available_ram_bytes"],
        "commit_headroom_bytes": launch["commit_headroom_bytes"], "disk_free_bytes": launch["disk_free_bytes"],
        "disk_target": launch["disk_target"], "artifact_cap_bytes": int(artifact_cap),
        "sampling_method": "live parent supervisor; psutil + GetPerformanceInfo + shutil.disk_usage"})
    if elapsed_cap_reached(clock() - attempt_started):
        raise TimeoutError("supervised cumulative or slice wall cap reached before child creation")
    stdout_path, stderr_path = report_dir / "replay.stdout.log", report_dir / "replay.stderr.log"
    started = attempt_started; proc = None; observed_process_identities: list[dict[str, Any]] = []
    tracked_execution_identity: dict[str, Any] | None = None
    stop_receipt = {"graceful_signal_sent": False,
        "forced_tree_termination": False, "stop_request_marker_status": "not_attempted",
        "stop_acknowledged": False}
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            proc = popen_factory(command, cwd=str(cwd), stdout=stdout, stderr=stderr, creationflags=creationflags)
            while True:
                now = clock()
                if elapsed_cap_reached(now - started):
                    raise TimeoutError("supervised cumulative or slice wall cap reached")
                code = proc.poll()
                sampled_pid, process_identities = executing_python_pid(int(proc.pid))
                if process_identities:
                    observed_process_identities = process_identities
                    for identity in process_identities:
                        if (expected_child_exe is not None
                                and os.path.normcase(identity.get("exe", "")) == os.path.normcase(expected_child_exe)):
                            tracked_execution_identity = identity
                for child_log in (stdout_path, stderr_path):
                    if child_log.stat().st_size > CHILD_LOG_CAP_BYTES:
                        raise TransitionError(f"child diagnostic log exceeded {CHILD_LOG_CAP_BYTES}-byte bound")
                if code is not None:
                    if tracked_execution_identity is not None:
                        try:
                            import psutil
                            executing = psutil.Process(int(tracked_execution_identity["pid"]))
                            if (float(executing.create_time()) == float(tracked_execution_identity["create_time"])
                                    and executing.is_running()):
                                raise TransitionError("Python child remains alive after its launcher exited")
                        except psutil.NoSuchProcess:
                            pass
                    for identity in observed_process_identities:
                        try:
                            import psutil
                            member = psutil.Process(int(identity["pid"]))
                            if (float(member.create_time()) == float(identity["create_time"])
                                    and member.is_running()):
                                raise TransitionError("owned process-tree member remains alive after launcher exit")
                        except psutil.NoSuchProcess:
                            continue
                    final = take_sample("child_exit_finalization", None)
                    final["attempt_id"] = attempt_id
                    final["elapsed_seconds"] = clock() - started
                    final_bytes = int(artifact_size_fn(output))
                    final["artifact_bytes"] = final_bytes
                    final_remaining = max(0, int(artifact_cap) - final_bytes)
                    final_failures = validate(final, launch=False, projected=final_remaining)
                    if final_bytes > int(artifact_cap):
                        final_failures.append({"metric":"artifact_bytes", "measured":final_bytes,
                            "maximum":int(artifact_cap)})
                    final["projected_remaining_write_bytes"] = final_remaining
                    final["runtime_gate_status"] = "failed" if final_failures else "passed"
                    sample_sequence += 1
                    _append_bounded(log, final, attempt_id=attempt_id, sequence=sample_sequence)
                    if final_failures:
                        raise TransitionError("finalization resource gate failed")
                    if elapsed_cap_reached(clock() - started):
                        raise TimeoutError("supervised cumulative or slice wall cap reached during final parent sampling")
                    result = {"status": "child_exited", "return_code": int(code), "pid": int(proc.pid),
                              "owned_process_identities": observed_process_identities,
                              "elapsed_seconds": clock() - started, "stop_receipt": stop_receipt,
                              "resource_log": str(log), "launch_receipt": str(launch_path)}
                    if code:
                        if child_receipt_path is not None and not Path(child_receipt_path).exists():
                            try:
                                stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4096:]
                                envelope_path = Path(child_receipt_path).parent / "private_child_envelope.json"
                                declared_parent = None
                                if envelope_path.is_file():
                                    child_envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
                                    secret = str(child_envelope.get("token", ""))
                                    if secret: stderr_tail = stderr_tail.replace(secret, "[redacted]")
                                    declared_parent = {"pid": child_envelope.get("parent_pid"),
                                        "create_time": child_envelope.get("parent_create_time"),
                                        "exe": child_envelope.get("parent_exe")}
                                lower_error = stderr_tail.lower()
                                ownership_reject = any(marker in lower_error for marker in (
                                    "private child ownership", "private child ancestry",
                                    "private child process ownership",
                                    "private child declared parent identity", "private child executable",
                                    "private child is not owned"))
                                _atomic_receipt(report_dir / "child_early_failure.json", {
                                    "attempt_id": attempt_id, "status": "failed_before_child_receipts",
                                    "phase": "private_child_ownership" if ownership_reject else "before_child_receipt_initialization",
                                    "owned_pid": int(proc.pid), "return_code": int(code),
                                    "declared_parent": declared_parent,
                                    "observed_process_identities": observed_process_identities,
                                    "child_receipt_status": "not_written",
                                    "stderr_tail": stderr_tail[-1200:],
                                    "model_construction": "not_attempted" if ownership_reject else "unknown",
                                    "optimizer_construction": "not_attempted" if ownership_reject else "unknown",
                                    "native_initialization": "not_attempted" if ownership_reject else "unknown"})
                            except Exception as evidence_error:
                                result["early_failure_evidence_error"] = repr(evidence_error)[:300]
                        failure = TransitionError(f"replay child exited with code {code}")
                        if result.get("early_failure_evidence_error"):
                            failure.add_note("early child failure evidence also failed: " +
                                str(result["early_failure_evidence_error"]))
                        raise failure
                    _atomic_receipt(report_dir / "owned_process_stop_receipt.json", result)
                    return result
                row = take_sample("runtime", sampled_pid)
                row["attempt_id"] = attempt_id
                if process_identities: row["owned_process_identities"] = process_identities
                row["elapsed_seconds"] = now - started
                artifact_bytes = int(artifact_size_fn(output))
                row["artifact_bytes"] = artifact_bytes
                row["projected_remaining_write_bytes"] = max(0, int(artifact_cap) - artifact_bytes)
                try:
                    failures = validate(row, launch=False, projected=row["projected_remaining_write_bytes"])
                    if artifact_bytes > int(artifact_cap):
                        failures.append({"metric":"artifact_bytes", "measured":artifact_bytes,
                            "maximum":int(artifact_cap)})
                    row["runtime_gate_status"] = "failed" if failures else "passed"
                    sample_sequence += 1
                    _append_bounded(log, row, attempt_id=attempt_id, sequence=sample_sequence)
                except Exception as log_exc:
                    try:
                        _atomic_receipt(report_dir / "supervisor_error.json", {"attempt_id": attempt_id,
                            "status": "resource_log_failure", "trigger_sample": row,
                            "error": repr(log_exc)[:500], "owned_pid": int(proc.pid)})
                    except Exception as receipt_exc:
                        log_exc.add_note(f"supervisor error receipt also failed: {receipt_exc!r}")
                    raise
                if failures:
                    failure = TransitionError(f"runtime resource gate failed: {failures}")
                    try:
                        _atomic_receipt(report_dir / "supervisor_error.json", {"attempt_id": attempt_id,
                            "status": "runtime_guard_failed", "trigger_sample": row,
                    "failures": failures, "owned_pid": int(proc.pid),
                    **child_failure_link(),
                    **({"native_loading_uncertainty": "child phase may have loaded native code",
                        "native_frame_uncertainty": "child phase progress at stop is not known to parent"}
                        if require_gpu else {})})
                    except Exception as receipt_exc:
                        failure.add_note(f"runtime guard receipt also failed: {receipt_exc!r}")
                    raise failure
                # Fixed delay after publication avoids catch-up bursts (and a
                # zero-sleep loop) if any bounded sample/write runs long.
                sleeper(float(sample_period))
    except BaseException as exc:
        # If the venv launcher exited after spawning Python, its child may have
        # been reparented. Stop only the PID/create-time identity observed in
        # this supervisor's own process tree.
        if tracked_execution_identity is not None and proc is not None:
            try:
                import psutil
                executing = psutil.Process(int(tracked_execution_identity["pid"]))
                if (float(executing.create_time()) == float(tracked_execution_identity["create_time"])
                        and executing.is_running()):
                    stop_receipt["graceful_executing_stop_sent"] = True
                    marker = report_dir / "STOP_REQUESTED"
                    try:
                        if not marker.exists():
                            with marker.open("xb") as stop_file:
                                stop_file.write(str(exc)[:1000].encode("utf-8"))
                                stop_file.flush(); os.fsync(stop_file.fileno())
                        stop_receipt["stop_request_marker_status"] = "written_and_fsynced"
                    except Exception as marker_error:
                        stop_receipt["stop_request_marker_status"] = "failed"
                        stop_receipt["executing_stop_marker_error"] = repr(marker_error)[:250]
                    try:
                        executing.wait(timeout=3)
                        stop_receipt["executing_process_stopped"] = True
                    except psutil.TimeoutExpired:
                        executing.terminate()
                        try: executing.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            executing.kill(); executing.wait(timeout=2)
                            stop_receipt["forced_tree_termination"] = True
                        stop_receipt["executing_process_stopped"] = True
                    stop_receipt["owned_process_tree"] = [int(executing.pid)]
            except Exception as owned_stop_error:
                stop_receipt["executing_process_stop_error"] = repr(owned_stop_error)[:300]
        if proc is not None:
            try:
                import psutil
                lingering = []
                for identity in observed_process_identities:
                    if int(identity["pid"]) == int(proc.pid): continue
                    try:
                        member = psutil.Process(int(identity["pid"]))
                        if (float(member.create_time()) == float(identity["create_time"])
                                and member.is_running()):
                            lingering.append(member)
                    except psutil.NoSuchProcess:
                        continue
                for member in lingering:
                    try: member.kill()
                    except psutil.NoSuchProcess: pass
                if lingering:
                    psutil.wait_procs(lingering, timeout=2)
                    stop_receipt["owned_process_tree"] = sorted(set(stop_receipt.get("owned_process_tree", [])) |
                        {int(member.pid) for member in lingering})
                    stop_receipt["forced_tree_termination"] = True
            except Exception as tree_cleanup_error:
                stop_receipt["owned_tree_cleanup_error"] = repr(tree_cleanup_error)[:300]
        if proc is not None and proc.poll() is None:
            stop_receipt["graceful_signal_sent"] = True
            stop_receipt["graceful_signal_status"] = "attempted"
            try:
                stop_receipt["stop_request_marker_status"] = "writing"
                stop_path = report_dir / "STOP_REQUESTED"
                if not stop_path.exists():
                    with stop_path.open("xb") as stop_file:
                        stop_file.write(str(exc)[:1000].encode("utf-8"))
                        stop_file.flush(); os.fsync(stop_file.fileno())
                stop_receipt["stop_request_marker_status"] = "written_and_fsynced"
                proc.wait(timeout=3)
                stop_receipt["stop_acknowledged"] = proc.poll() is not None
            except Exception as graceful_error:
                if stop_receipt["stop_request_marker_status"] != "written_and_fsynced":
                    stop_receipt["stop_request_marker_status"] = "failed"
                stop_receipt["graceful_stop_error"] = repr(graceful_error)[:400]
                try:
                    if os.name == "nt":
                        proc.send_signal(getattr(__import__("signal"), "CTRL_BREAK_EVENT"))
                    else:
                        proc.terminate()
                    proc.wait(timeout=3)
                    stop_receipt["stop_acknowledged"] = proc.poll() is not None
                except Exception as stop_exc:
                    try:
                        import psutil
                        root = psutil.Process(int(proc.pid))
                        children = root.children(recursive=True)
                        stop_receipt["owned_process_tree"] = [p.pid for p in [*children, root]]
                        for owned in [*children, root]:
                            try: owned.kill()
                            except psutil.NoSuchProcess: pass
                        psutil.wait_procs([*children, root], timeout=3)
                        stop_receipt["forced_tree_termination"] = True
                        stop_receipt["stop_acknowledged"] = proc.poll() is not None
                    except Exception as tree_exc:
                        stop_receipt["stop_error"] = repr(tree_exc)[:500]
        try:
            exit_code = None
            if proc is not None:
                try: exit_code = proc.poll()
                except Exception as poll_error: stop_receipt["post_stop_poll_error"] = repr(poll_error)[:300]
            confirmed_stopped = exit_code is not None or bool(stop_receipt.get("stop_acknowledged"))
            _atomic_receipt(report_dir / "owned_process_stop_receipt.json", {"attempt_id": attempt_id,
                "status": "stopped_after_failure" if confirmed_stopped else "stop_unconfirmed_after_failure",
                "confirmed_stopped": confirmed_stopped, "primary_error": repr(exc)[:1000],
                "pid": int(proc.pid) if proc is not None else None,
                "elapsed_seconds": clock() - started, "child_exit_code_after_stop": exit_code,
                "observed_process_identities": observed_process_identities,
                **child_failure_link(), **stop_receipt})
        except Exception as receipt_exc:
            exc.add_note(f"owned-child stop receipt also failed: {receipt_exc!r}")
        raise


def _atomic_receipt(path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable receipt; occupied paths and write errors are fatal."""
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if len(data) > IMMUTABLE_RECEIPT_CAP_BYTES:
        raise TransitionError("immutable lifecycle receipt exceeds 64 KiB bound")
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())

