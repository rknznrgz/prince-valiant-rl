"""Bounded evaluation traces and exact policy-input fingerprints."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping

import numpy as np


SERIALIZATION_VERSION = "numpy-contiguous-concatenation-v1"
COMPARISON_SCOPE = "exact encoded model input fields, in model_window_keys order"
DEFAULT_RESERVE_BYTES = 1024 * 1024 * 1024
DEFAULT_ARTIFACT_BUDGET_BYTES = 500 * 1024 * 1024
DEFAULT_TEMP_OVERHEAD_BYTES = 1024 * 1024
DEFAULT_FLUSH_ROWS = 128


def atomic_write_json(path: Path | str, payload: Any, *, replace_retries: int = 40,
                      retry_delay_seconds: float = 0.05) -> None:
    """Atomically replace a small JSON file after closing its unique temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="\n", delete=False,
                dir=target.parent, prefix=f"{target.name}.", suffix=".pending") as stream:
            pending = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(max(0, int(replace_retries)) + 1):
            try:
                os.replace(pending, target)
                pending = None
                return
            except PermissionError:
                if attempt >= int(replace_retries):
                    raise
                time.sleep(max(0.0, float(retry_delay_seconds)))
    finally:
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass


def _canonical_arrays(observation: Mapping[str, Any], keys: Iterable[str]):
    for key in keys:
        array = np.ascontiguousarray(observation[key])
        yield key, array, array.tobytes()


def policy_input_schema(observation: Mapping[str, Any], keys: Iterable[str]) -> dict:
    """Describe the exact contiguous-array serialization used by saved inputs."""
    arrays = []
    offset = 0
    ordered_keys = list(keys)
    for key, array, payload in _canonical_arrays(observation, ordered_keys):
        arrays.append({"key": key, "dtype": array.dtype.str,
                       "shape": list(array.shape), "byte_offset": offset,
                       "byte_length": len(payload)})
        offset += len(payload)
    return {"serialization_version": SERIALIZATION_VERSION,
            "comparison_scope": COMPARISON_SCOPE,
            "model_window_keys": ordered_keys,
            "arrays": arrays, "record_bytes": offset}


def fingerprint_policy_input(observation: Mapping[str, Any], keys: Iterable[str]) -> dict:
    """Hash each field and the concatenated canonical record without joining fields."""
    ordered_keys = list(keys)
    record_digest = hashlib.sha256()
    fields = {}
    for key, array, payload in _canonical_arrays(observation, ordered_keys):
        record_digest.update(payload)
        fields[key] = {"dtype": array.dtype.str, "shape": list(array.shape),
                       "sha256": hashlib.sha256(payload).hexdigest()}
    return {"serialization_version": SERIALIZATION_VERSION,
            "comparison_scope": COMPARISON_SCOPE,
            "record_sha256": record_digest.hexdigest(), "fields": fields}


def fingerprint_reference_record(payload: bytes, schema: Mapping[str, Any]) -> dict:
    """Fingerprint one record in the saved-reference concatenated byte format."""
    record_bytes = int(schema["record_bytes"])
    if len(payload) != record_bytes:
        raise ValueError(f"Reference record has {len(payload)} bytes; expected {record_bytes}")
    fields = {}
    for spec in schema["arrays"]:
        start = int(spec["byte_offset"])
        stop = start + int(spec["byte_length"])
        if start < 0 or stop > record_bytes:
            raise ValueError("Reference field extends outside its canonical record")
        fields[spec["key"]] = {"dtype": str(spec["dtype"]),
                                "shape": list(spec["shape"]),
                                "sha256": hashlib.sha256(payload[start:stop]).hexdigest()}
    if list(fields) != list(schema["model_window_keys"]):
        raise ValueError("Reference field order differs from model_window_keys")
    return {"serialization_version": SERIALIZATION_VERSION,
            "comparison_scope": COMPARISON_SCOPE,
            "record_sha256": hashlib.sha256(payload).hexdigest(), "fields": fields}


def fingerprints_equal(actual: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    """Exact match requires the same serialization, field metadata, and field hashes."""
    return (actual.get("serialization_version") == SERIALIZATION_VERSION
            and reference.get("serialization_version") == SERIALIZATION_VERSION
            and actual.get("comparison_scope") == COMPARISON_SCOPE
            and reference.get("comparison_scope") == COMPARISON_SCOPE
            and actual.get("record_sha256") == reference.get("record_sha256")
            and actual.get("fields") == reference.get("fields"))


class BoundedJsonlWriter:
    """Write bounded JSONL chunks; compressed output uses closed gzip members."""
    def __init__(self, path: Path, *, compressed: bool = True,
                 flush_rows: int = DEFAULT_FLUSH_ROWS, max_line_bytes: int = 1024 * 1024):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = self.path.open("wb", buffering=0)
        self.compressed = bool(compressed)
        self.flush_rows = max(1, int(flush_rows))
        self.max_line_bytes = int(max_line_bytes)
        self.stream = None
        if self.compressed:
            self._start_member()
        else:
            self.stream = self.raw
        self.rows_written = 0
        self.uncompressed_bytes = 0
        self.max_line_bytes_seen = 0
        self.rows_since_flush = 0
        self.closed = False

    def __enter__(self):
        return self

    def _start_member(self) -> None:
        self.stream = gzip.GzipFile(filename="", mode="wb", compresslevel=1,
                                    fileobj=self.raw, mtime=0)

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    @property
    def compressed_bytes(self) -> int:
        try:
            return int(self.raw.tell())
        except (OSError, ValueError):
            return 0

    def write(self, row: Mapping[str, Any]) -> None:
        if self.closed:
            raise ValueError("Cannot write to a closed evaluation capture")
        if self.compressed and self.stream is None:
            self._start_member()
        encoded = (json.dumps(row, separators=(",", ":"), allow_nan=False,
                              ensure_ascii=False) + "\n").encode("utf-8")
        if len(encoded) > self.max_line_bytes:
            raise ValueError(f"Evaluation capture row exceeds {self.max_line_bytes} bytes")
        self.stream.write(encoded)
        self.rows_written += 1
        self.uncompressed_bytes += len(encoded)
        self.max_line_bytes_seen = max(self.max_line_bytes_seen, len(encoded))
        self.rows_since_flush += 1
        if self.rows_since_flush >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if self.closed:
            return
        if self.compressed:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            self.raw.flush()
        else:
            self.raw.flush()
        self.rows_since_flush = 0

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.compressed:
                if self.stream is not None:
                    self.stream.close()
                    self.stream = None
                self.raw.flush()
            else:
                self.raw.flush()
        finally:
            try:
                self.raw.close()
            finally:
                self.closed = True


class StorageGuardInterruption(RuntimeError):
    """Raised before another environment transition would violate free space."""
    def __init__(self, minimum_free_bytes: int, measurements: list[dict]):
        self.minimum_free_bytes = int(minimum_free_bytes)
        self.measurements = measurements
        minimum = min((row["free_bytes"] for row in measurements), default=None)
        super().__init__(f"disk guard: minimum free space is {minimum_free_bytes} bytes; "
                         f"measured minimum is {minimum} bytes")


class ArtifactBudgetInterruption(RuntimeError):
    """Raised with a small finalization margin before the artifact cap is reached."""
    def __init__(self, used_bytes: int, budget_bytes: int, safety_margin_bytes: int):
        self.used_bytes = int(used_bytes)
        self.budget_bytes = int(budget_bytes)
        self.safety_margin_bytes = int(safety_margin_bytes)
        super().__init__(f"artifact budget guard: {used_bytes} bytes used; "
                         f"budget {budget_bytes} bytes with {safety_margin_bytes} bytes reserved")


class ArtifactBudgetGuard:
    """Bound only the explicitly supplied, task-owned output tree."""
    def __init__(self, root: Path | str, *, maximum_bytes: int,
                 check_every: int = 64, safety_margin_bytes: int = 2 * 1024 * 1024):
        self.root = Path(root)
        self.maximum_bytes = int(maximum_bytes)
        self.check_every = max(1, int(check_every))
        self.safety_margin_bytes = int(safety_margin_bytes)
        self.checks = 0
        self.last_used_bytes = self._tree_bytes()

    def _tree_bytes(self) -> int:
        if not self.root.exists():
            return 0
        total = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                # Atomic status publication may remove a sibling pending file
                # after rglob/is_file sees it. The configured margin covers
                # this short scan race; a later check sees the final file.
                continue
        return total

    def check(self, *, force: bool = False) -> int:
        self.checks += 1
        if force or self.checks % self.check_every == 0:
            self.last_used_bytes = self._tree_bytes()
        if self.last_used_bytes + self.safety_margin_bytes > self.maximum_bytes:
            raise ArtifactBudgetInterruption(self.last_used_bytes,
                                             self.maximum_bytes,
                                             self.safety_margin_bytes)
        return self.last_used_bytes


def disk_measurements(paths: Iterable[Path | str]) -> list[dict]:
    """Measure each distinct output/temp volume without scanning its contents."""
    measurements = []
    seen = set()
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            candidate = candidate.parent
        anchor = os.path.abspath(candidate).split(os.sep)[0].casefold()
        try:
            usage = shutil.disk_usage(candidate)
        except OSError as exc:
            measurements.append({"path": str(candidate), "status": "unavailable",
                                 "error": repr(exc), "free_bytes": None})
            continue
        key = (anchor, int(usage.free))
        if key in seen:
            continue
        seen.add(key)
        measurements.append({"path": str(candidate), "status": "available",
                             "total_bytes": int(usage.total),
                             "free_bytes": int(usage.free),
                             "used_bytes": int(usage.used)})
    return measurements


class DiskGuard:
    def __init__(self, paths: Iterable[Path | str], *, minimum_free_bytes: int = DEFAULT_RESERVE_BYTES):
        self.paths = tuple(Path(path) for path in paths)
        self.minimum_free_bytes = int(minimum_free_bytes)
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes must be nonnegative")

    def check(self) -> list[dict]:
        measurements = disk_measurements(self.paths)
        if not measurements or any(row["status"] != "available" for row in measurements):
            raise StorageGuardInterruption(self.minimum_free_bytes, measurements)
        if min(row["free_bytes"] for row in measurements) < self.minimum_free_bytes:
            raise StorageGuardInterruption(self.minimum_free_bytes, measurements)
        return measurements


class ReferenceFingerprintReader:
    """Read one precomputed reference fingerprint per policy decision."""
    def __init__(self, data_path: Path | str, *, expected_record_count: int):
        self.path = Path(data_path)
        self.stream = gzip.open(self.path, "rt", encoding="utf-8")
        self.expected_record_count = int(expected_record_count)
        self.read_count = 0
        self.closed = False

    def read(self, decision_index: int) -> dict | None:
        line = self.stream.readline()
        if not line:
            if self.read_count != self.expected_record_count:
                raise ValueError("Reference fingerprint stream ended before its manifest count")
            return None
        row = json.loads(line)
        if int(row.get("decision_index", -1)) != int(decision_index):
            raise ValueError("Reference fingerprint decision indices are not contiguous")
        self.read_count += 1
        return row

    def close(self) -> None:
        if not self.closed:
            self.stream.close()
            self.closed = True


def assess_storage(paths: Iterable[Path | str], *, artifact_budget_bytes: int,
                   reserve_bytes: int = DEFAULT_RESERVE_BYTES,
                   temp_overhead_bytes: int = DEFAULT_TEMP_OVERHEAD_BYTES) -> dict:
    """Check the required initial reserve + new-artifact budget + temp allowance."""
    measurements = disk_measurements(paths)
    required = int(reserve_bytes) + int(artifact_budget_bytes) + int(temp_overhead_bytes)
    available = (min(row["free_bytes"] for row in measurements)
                 if measurements and all(row["status"] == "available" for row in measurements)
                 else None)
    return {"status": "passed" if available is not None and available >= required else "insufficient",
            "available_free_bytes": available,
            "required_free_bytes": required,
            "reserve_bytes": int(reserve_bytes),
            "artifact_budget_bytes": int(artifact_budget_bytes),
            "temporary_overhead_bytes": int(temp_overhead_bytes),
            "shortfall_bytes": None if available is None else max(0, required - available),
            "volumes": measurements}


def partial_capture_status(*, frames: int, completion: Any, terminal_reason: str,
                           interruption: str, measurements: list[dict] | None = None) -> dict:
    """Represent interruption without presenting it as a terminal outcome."""
    return {"assessment_status": "partial_assessment",
            "native_frames": int(frames),
            "completion": None,
            "natural_terminal": False,
            "terminal_reason": str(terminal_reason),
            "last_observed_completion": completion,
            "interruption": str(interruption),
            "validity": "partial_valid" if frames else "unavailable",
            "storage_measurements": measurements}


def iter_json_array(path: Path | str, *, chunk_chars: int = 256 * 1024) -> Iterator[Any]:
    """Iterate a top-level JSON array without materializing the complete array."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as stream:
        buffer = ""
        ended = False

        def fill() -> bool:
            nonlocal buffer, ended
            if ended:
                return False
            chunk = stream.read(chunk_chars)
            if not chunk:
                ended = True
                return False
            buffer += chunk
            return True

        while not buffer and fill():
            pass
        pos = 0
        while pos < len(buffer) and buffer[pos].isspace():
            pos += 1
        if pos >= len(buffer) and not fill():
            raise ValueError("JSON trace is empty")
        if buffer[pos] != "[":
            raise ValueError("Saved source trace must be a JSON array")
        buffer = buffer[pos + 1:]
        need_value = True
        while True:
            while True:
                buffer = buffer.lstrip()
                if buffer and (buffer[0] == "," and not need_value):
                    buffer = buffer[1:]
                    need_value = True
                    continue
                if buffer:
                    break
                if not fill():
                    raise ValueError("Saved source trace is truncated")
            if buffer[0] == "]":
                buffer = buffer[1:]
                if buffer.strip() or stream.read().strip():
                    raise ValueError("Unexpected trailing data after saved source trace")
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if not fill():
                    raise
                continue
            yield value
            buffer = buffer[end:]
            need_value = False


def build_reference_fingerprint_bundle(*, trace_path: Path | str, input_path: Path | str,
        input_schema: Mapping[str, Any], output_path: Path | str, metadata_path: Path | str,
        expected_trace_sha256: str, expected_input_sha256: str,
        expected_schema_sha256: str | None = None) -> dict:
    """Build a compact reference stream from verified trace and saved policy inputs."""
    trace_path, input_path = Path(trace_path), Path(input_path)
    output_path, metadata_path = Path(output_path), Path(metadata_path)

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    schema_digest = hashlib.sha256(json.dumps(input_schema, sort_keys=True,
                                               separators=(",", ":")).encode()).hexdigest()
    if expected_schema_sha256 and schema_digest != expected_schema_sha256:
        raise ValueError("Reference policy-input schema fingerprint does not match")
    if file_sha256(trace_path) != expected_trace_sha256:
        raise ValueError("Reference trace hash does not match its verified source")
    if file_sha256(input_path) != expected_input_sha256:
        raise ValueError("Reference policy-input hash does not match its verified source")
    input_size = int(input_path.stat().st_size)
    record_bytes = int(input_schema["record_bytes"])
    if record_bytes <= 0 or input_size % record_bytes:
        raise ValueError("Reference policy-input stream has an invalid length")
    record_count = input_size // record_bytes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending_path = output_path.with_suffix(output_path.suffix + ".partial")
    rows = 0
    with input_path.open("rb") as inputs:
        as_writer = BoundedJsonlWriter(pending_path)
        try:
            for index, trace_row in enumerate(iter_json_array(trace_path)):
                if index >= record_count:
                    raise ValueError("Reference trace has more rows than saved policy inputs")
                payload = inputs.read(record_bytes)
                if len(payload) != record_bytes:
                    raise ValueError("Reference policy-input stream ended before its trace")
                fingerprint = fingerprint_reference_record(payload, input_schema)
                as_writer.write({"decision_index": index,
                                 "action": int(trace_row["action"]),
                                 "mask": int(trace_row["mask"]),
                                 "action_probabilities": trace_row.get("action_probabilities"),
                                 "ram_sha256": trace_row.get("ram_sha256"),
                                 "input": fingerprint})
                rows += 1
            if rows != record_count or inputs.read(1):
                raise ValueError("Reference trace and saved policy-input records are not aligned")
        finally:
            as_writer.close()
    pending_path.replace(output_path)
    metadata = {"status": "verified",
                "format_version": "level1-policy-input-fingerprints-v1",
                "serialization_version": SERIALIZATION_VERSION,
                "comparison_scope": COMPARISON_SCOPE,
                "source_trace_sha256": expected_trace_sha256,
                "source_policy_inputs_sha256": expected_input_sha256,
                "source_policy_input_schema_sha256": schema_digest,
                "source_trace_path": str(trace_path),
                "source_policy_inputs_path": str(input_path),
                "record_count": rows, "record_bytes": record_bytes,
                "data_path": str(output_path),
                "data_sha256": file_sha256(output_path),
                "data_bytes": output_path.stat().st_size}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".pending")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return metadata
