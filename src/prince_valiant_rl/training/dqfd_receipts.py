"""Bounded append-only receipt journals for one-writer DQfD processes."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_RECORD_BYTES = 64 * 1024
MAX_JOURNAL_BYTES = 12 * 1024 * 1024
MAX_RECORDS = 32768


def _fsync_receipt(path: Path, stream: Any) -> None:
    """Durable receipt boundary, split out so offline fault tests can fail the real writer."""
    os.fsync(stream.fileno())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def serialize_record(payload: dict[str, Any], *, attempt_id: str, sequence: int,
                     written_at_utc: str | None = None) -> tuple[dict[str, Any], bytes]:
    """Use the production canonical serializer and include the newline framing byte."""
    body = {"attempt_id": str(attempt_id), "sequence": int(sequence),
            "written_at_utc": written_at_utc or datetime.now(timezone.utc).isoformat(),
            "payload": payload}
    envelope = {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}
    return envelope, _canonical(envelope) + b"\n"


def append_record(path: Path, payload: dict[str, Any], *, attempt_id: str, sequence: int,
                  max_bytes: int = MAX_JOURNAL_BYTES) -> dict[str, Any]:
    """Append and fsync one checksummed record; write errors are fatal."""
    path = Path(path)
    envelope, data = serialize_record(payload, attempt_id=attempt_id, sequence=sequence)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("receipt record exceeds 64 KiB")
    current = path.stat().st_size if path.exists() else 0
    if current + len(data) > max_bytes:
        raise ValueError("receipt journal reached its bounded size")
    with path.open("ab") as stream:
        written = stream.write(data)
        if written != len(data):
            raise OSError("short receipt journal write")
        stream.flush()
        _fsync_receipt(path, stream)
    return envelope


def _parse_records(data: bytes, *, attempt_id: str, max_bytes: int,
                   allow_truncated_tail: bool) -> list[dict[str, Any]]:
    if len(data) > max_bytes:
        raise ValueError("receipt journal exceeds scan bound")
    if data and not data.endswith(b"\n"):
        if not allow_truncated_tail:
            raise ValueError("receipt journal has an incomplete tail")
        data = data[:data.rfind(b"\n") + 1]
    lines = data.splitlines()
    if len(lines) > MAX_RECORDS:
        raise ValueError("receipt journal exceeds record scan bound")
    records: list[dict[str, Any]] = []
    expected = 1
    for line in lines:
        if not line or len(line) + 1 > MAX_RECORD_BYTES:
            raise ValueError("malformed receipt record size")
        envelope = json.loads(line)
        body = {k: v for k, v in envelope.items() if k != "sha256"}
        if envelope.get("attempt_id") != attempt_id:
            raise ValueError("receipt journal attempt id mismatch")
        if int(envelope.get("sequence", -1)) != expected:
            raise ValueError("receipt journal sequence mismatch")
        if envelope.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest():
            raise ValueError("receipt journal checksum mismatch")
        records.append(envelope)
        expected += 1
    return records


def read_records(path: Path, *, attempt_id: str, max_bytes: int = MAX_JOURNAL_BYTES,
                 allow_truncated_tail: bool = True) -> list[dict[str, Any]]:
    """Read bounded valid records; only an incomplete final line may be ignored."""
    return _parse_records(Path(path).read_bytes(), attempt_id=attempt_id,
        max_bytes=max_bytes, allow_truncated_tail=allow_truncated_tail)


def latest_record(path: Path, *, attempt_id: str) -> dict[str, Any]:
    """Validate only the bounded tail for live readers; a partial tail is skipped."""
    path = Path(path)
    size = path.stat().st_size
    if size > MAX_JOURNAL_BYTES:
        raise ValueError("receipt journal exceeds scan bound")
    with path.open("rb") as stream:
        stream.seek(max(0, size - (2 * MAX_RECORD_BYTES + 1)))
        tail = stream.read()
    lines = tail.split(b"\n")
    if tail and not tail.endswith(b"\n"):
        lines.pop()  # incomplete final record never becomes a fresh sample
    candidates = [line for line in lines if line]
    if not candidates:
        raise ValueError("receipt journal has no complete valid records")
    line = candidates[-1]
    if len(line) + 1 > MAX_RECORD_BYTES:
        raise ValueError("receipt tail record exceeds 64 KiB bound")
    envelope = json.loads(line)
    body = {k: v for k, v in envelope.items() if k != "sha256"}
    if envelope.get("attempt_id") != attempt_id:
        raise ValueError("receipt journal attempt id mismatch")
    if int(envelope.get("sequence", 0)) < 1:
        raise ValueError("receipt journal sequence invalid")
    if envelope.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("receipt journal checksum mismatch")
    return envelope


class IncrementalRecordReader:
    """Verify a bounded append-only journal once, then validate only new lines.

    The initial pass has exactly ``read_records`` semantics. Later passes reject
    truncation, incomplete records (which remain pending), sequence/attempt
    drift, malformed rows and checksum failures without rescanning old bytes.
    """
    def __init__(self, path: Path, *, attempt_id: str,
                 max_bytes: int = MAX_JOURNAL_BYTES) -> None:
        self.path = Path(path)
        self.attempt_id = str(attempt_id)
        self.max_bytes = int(max_bytes)
        raw = self.path.read_bytes()
        records = _parse_records(raw, attempt_id=self.attempt_id,
            max_bytes=self.max_bytes, allow_truncated_tail=True)
        self.latest = records[-1] if records else None
        self.sequence = len(records) + 1
        self.offset = len(raw) if not raw or raw.endswith(b"\n") else raw.rfind(b"\n") + 1
        self.pending = raw[self.offset:]

    def refresh(self) -> dict[str, Any] | None:
        size = self.path.stat().st_size
        if size > self.max_bytes:
            raise ValueError("receipt journal exceeds scan bound")
        if size < self.offset + len(self.pending):
            raise ValueError("receipt journal was truncated during incremental read")
        with self.path.open("rb") as stream:
            stream.seek(self.offset + len(self.pending))
            added = stream.read()
        data = self.pending + added
        if data and not data.endswith(b"\n"):
            split = data.rfind(b"\n") + 1
            complete, self.pending = data[:split], data[split:]
        else:
            complete, self.pending = data, b""
        self.offset += len(complete)
        for line in complete.splitlines():
            if self.sequence > MAX_RECORDS:
                raise ValueError("receipt journal exceeds record scan bound")
            if not line or len(line) + 1 > MAX_RECORD_BYTES:
                raise ValueError("malformed receipt record size")
            envelope = json.loads(line)
            body = {k: v for k, v in envelope.items() if k != "sha256"}
            if envelope.get("attempt_id") != self.attempt_id:
                raise ValueError("receipt journal attempt id mismatch")
            if int(envelope.get("sequence", -1)) != self.sequence:
                raise ValueError("receipt journal sequence mismatch")
            if envelope.get("sha256") != hashlib.sha256(_canonical(body)).hexdigest():
                raise ValueError("receipt journal checksum mismatch")
            self.latest = envelope
            self.sequence += 1
        return self.latest
