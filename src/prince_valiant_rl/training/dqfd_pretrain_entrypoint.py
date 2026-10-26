"""Bounded public-parent/private-child wiring for the frozen DQfD pretrain slice."""
from __future__ import annotations

import hashlib, hmac, json, math, os, secrets, sys, time, uuid
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable

from . import dqfd_production as production
from .dqfd_pretraining import run_demo_pretraining
from .dqfd_receipts import IncrementalRecordReader, append_record, read_records
from .dqfd_supervision import supervise_replay, validate_ml_resource_sample
from .dqfd_process_ownership import process_record as _process_record
from .dqfd_process_ownership import validate_private_child_ownership
from .dqfd_study_budget import SCHEMA as EXTENDED_BUDGET_SCHEMA, validate_extended_receipt, caps_for_budget

UPDATES = 2000
MODEL_IO_CAP = 600.0
DEMO_CAP = 900.0
SLICE_CAP = 1500.0
RESOURCE_SAMPLE_PERIOD_SECONDS = 0.5
TICKS_PER_SECOND = 10_000_000  # exact 100 ns integer accounting units
BASE_PROTOCOL = MappingProxyType({"original_total_active_seconds": 5400,
    "model_io_active_seconds": 600, "demo_encoding_active_seconds": 900,
    "slice_active_seconds": 1500, "updates": 2000, "batch_size": 128,
    "beta": .6, "seed": 20261024})
BASE_PROTOCOL_SHA256 = hashlib.sha256(json.dumps(dict(BASE_PROTOCOL), sort_keys=True,
    separators=(",", ":"), allow_nan=False).encode()).hexdigest()
PROMPT11_SHA256 = "74a7f1744688c7c7c6b6f57c059a266f8329d0d14a08c2544163552e9a4880ed"
# Compatibility name for frozen acceptance fixtures only; it contains no
# current prior charge or launch-ready remainder.
EXPECTED_BUDGET = BASE_PROTOCOL
MIN_KNOWN_PRIOR_TICKS = 4_401_807_673  # stage18 live charge, cannot regress below this receipt
PINNED_PREDECESSOR_HASHES = MappingProxyType({
    "stage18_timing_receipt_sha256": "3b696905f5b63493e8432aa6971abaddcd9179269a2edd0844d79d191f7cc216",
    "stage19_acceptance_result_sha256": "fe347c43fbf13dc2c9574d1149a60824c10b3e06d03eb18e88d01aa3314f7175",
    "stage20_result_sha256": "8748bc4ec6aa5313cfc72c92b68666b8c9ca512c58f0c2f0948dbf4a46a4af88",
    "stage20_budget_amendment_sha256": "34f9ce56158668a5b27d9d612fcc86b45a199b924d52f6fe20e7ca2756bc526e"})
EXECUTION_BUDGET_RECEIPT_FIELDS = frozenset({"schema", "base_protocol_sha256", "protocol_sha256",
    "predecessor_receipt_sha256", "accounting_basis", "original_total_ticks", "prior_charged_ticks",
    "remaining_ticks", "known_supervised_live_charge_ticks", "measured_prelaunch_ticks",
    "measured_prelaunch_included", "additional_explicit_charge_ticks", "additional_explicit_charge_basis",
    "unmeasured_historical_review_time", "receipt_sha256"})
EXPECTED_CAPS = {"episode_count_max": 3, "episode_native_frames_max": 36000,
    "demo_native_frames_total_max": 108000, "video_bytes_each_max": 104857600,
    "video_bytes_total_max": 314572800, "whole_pilot_native_frames_remaining": 684640,
    "retained_bytes_total_max": 12884901888}
CODE_PATHS = ("scripts/level1_dqfd_pilot.py",
    "src/prince_valiant_rl/training/dqfd_production.py",
    "src/prince_valiant_rl/training/dqfd_pretraining.py",
    "src/prince_valiant_rl/training/dqfd_pretrain_entrypoint.py",
    "src/prince_valiant_rl/training/dqfd_process_ownership.py",
    "src/prince_valiant_rl/training/dqfd_evaluation.py",
    "src/prince_valiant_rl/training/dqfd_execution.py",
    "src/prince_valiant_rl/training/dqfd_supervision.py",
    "src/prince_valiant_rl/training/dqfd_level1_adapter.py",
    "src/prince_valiant_rl/training/dqfd.py",
    "src/prince_valiant_rl/training/dqfd_replay.py",
    "src/prince_valiant_rl/training/dqfd_transitions.py",
    "src/prince_valiant_rl/training/dqfd_receipts.py",
    "src/prince_valiant_rl/training/temporal_models.py",
    "src/prince_valiant_rl/training/models.py",
    "src/prince_valiant_rl/training/temporal.py",
    "src/prince_valiant_rl/backward_level1.py",
    "src/prince_valiant_rl/recorder_native.py",
    "src/prince_valiant_rl/control_history.py",
    "src/prince_valiant_rl/route_guidance.py",
    "src/prince_valiant_rl/route_waypoints.py",
    "src/prince_valiant_rl/navigation.py",
    "src/prince_valiant_rl/game_state.py",
    "src/prince_valiant_rl/kind18_events.py",
    "src/prince_valiant_rl/training/dqfd_study_budget.py",
    "src/prince_valiant_rl/training/dqfd_continuation.py")


def production_code_hash(root: Path) -> str:
    digests = [hashlib.sha256((Path(root) / relative).read_bytes()).hexdigest() for relative in CODE_PATHS]
    return hashlib.sha256("".join(digests).encode()).hexdigest()


def _counter_snapshot(learner: Any) -> dict[str, Any]:
    """Copy counters before cleanup; unavailable values remain explicitly unknown."""
    counters = getattr(learner, "counters", None) if learner is not None else None
    names = ("attempted_steps", "applied_steps", "successful_steps", "failed_after_step",
             "unknown_step_outcomes", "target_copies")
    return {name: (int(value) if isinstance((value := getattr(counters, name, None)), (int, bool)) else None)
            for name in names}


def _elapsed_ticks(elapsed: float) -> int:
    value = float(elapsed)
    if not math.isfinite(value) or value < 0:
        raise TimeoutError("invalid supervised elapsed time")
    return math.ceil(value * TICKS_PER_SECOND)


def check_original_total_deadline(elapsed: float, budget: dict[str, Any]) -> float:
    """Use integer 100 ns ticks; ceil elapsed so conversion never grants time."""
    elapsed_ticks = _elapsed_ticks(elapsed)
    if elapsed_ticks > int(budget["remaining_active_ticks"]):
        raise TimeoutError("original cumulative active-time guard blocks child work")
    if elapsed_ticks > int(budget["slice_active_ticks"]):
        raise TimeoutError("supervised slice active-time guard blocks child work")
    return (int(budget["prior_charged_ticks"]) + elapsed_ticks) / TICKS_PER_SECOND


def _canonical_receipt_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def validate_execution_budget_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate the caller-selected hash-linked ledger; no report-tree discovery."""
    if isinstance(receipt, dict) and receipt.get("schema") == "dqfd_demo_continuation_budget_v3":
        from .dqfd_continuation import validate_budget
        return validate_budget(receipt, PROMPT11_SHA256)
    if isinstance(receipt, dict) and receipt.get("schema") == EXTENDED_BUDGET_SCHEMA:
        return validate_extended_receipt(receipt, protocol_sha256=PROMPT11_SHA256)
    if not isinstance(receipt, dict) or receipt.get("schema") != "dqfd_current_execution_budget_v1":
        raise ValueError("execution budget receipt schema is missing or foreign")
    if set(receipt) != EXECUTION_BUDGET_RECEIPT_FIELDS:
        raise ValueError("execution budget receipt fields are incomplete or unrecognized")
    if receipt.get("base_protocol_sha256") != BASE_PROTOCOL_SHA256:
        raise ValueError("execution budget base protocol hash mismatch")
    if receipt.get("protocol_sha256") != PROMPT11_SHA256:
        raise ValueError("execution budget original protocol SHA mismatch")
    if receipt.get("predecessor_receipt_sha256") != dict(PINNED_PREDECESSOR_HASHES):
        raise ValueError("execution budget predecessor linkage mismatch")
    if receipt.get("receipt_sha256") != _canonical_receipt_hash(receipt):
        raise ValueError("execution budget receipt hash mismatch")
    values = (receipt.get("original_total_ticks"), receipt.get("prior_charged_ticks"),
        receipt.get("remaining_ticks"), receipt.get("known_supervised_live_charge_ticks"),
        receipt.get("measured_prelaunch_ticks"), receipt.get("additional_explicit_charge_ticks"))
    if any(type(value) is not int for value in values):
        raise ValueError("execution budget ticks must be finite nonnegative integers")
    total, prior, remaining, known_live, prelaunch, additional = values
    if any(value < 0 for value in values):
        raise ValueError("execution budget tick values must be nonnegative")
    if total != 5400 * TICKS_PER_SECOND or not 0 <= prior < total:
        raise ValueError("execution budget original cap or prior charge is invalid")
    if prior < MIN_KNOWN_PRIOR_TICKS:
        raise ValueError("execution budget prior charge regresses below pinned stage18 usage")
    if known_live != MIN_KNOWN_PRIOR_TICKS or prelaunch != 7 * TICKS_PER_SECOND:
        raise ValueError("execution budget accounting components differ from preserved stage18/stage20 evidence")
    included_prelaunch = receipt.get("measured_prelaunch_included")
    expected_prior = known_live + (prelaunch if included_prelaunch is True else 0) + additional
    if type(included_prelaunch) is not bool or prior != expected_prior:
        raise ValueError("execution budget prior does not reconcile to its explicit accounting components")
    charge_basis = receipt.get("additional_explicit_charge_basis")
    if (not isinstance(charge_basis, str)
            or (additional == 0 and charge_basis != "none")
            or (additional > 0 and not charge_basis.strip())
            or (additional > 0 and charge_basis == "none")):
        raise ValueError("additional budget deduction requires an explicit nonempty authorization basis")
    if remaining != total - prior or remaining < 0:
        raise ValueError("execution budget remainder is inconsistent")
    if not isinstance(receipt.get("accounting_basis"), str) or not receipt["accounting_basis"].strip():
        raise ValueError("execution budget accounting basis is missing")
    values = dict(BASE_PROTOCOL)
    values.update(prior_charged_active_seconds=f"{prior / TICKS_PER_SECOND:.7f}",
        remaining_original_active_seconds=f"{remaining / TICKS_PER_SECOND:.7f}",
        remaining_seconds_deadline=math.nextafter(remaining / TICKS_PER_SECOND, -math.inf),
        prior_charged_ticks=prior, remaining_active_ticks=remaining,
        slice_active_ticks=1500 * TICKS_PER_SECOND,
        model_io_active_ticks=600 * TICKS_PER_SECOND,
        demo_encoding_active_ticks=900 * TICKS_PER_SECOND,
        base_protocol_sha256=BASE_PROTOCOL_SHA256,
        execution_budget_receipt_sha256=receipt["receipt_sha256"])
    return values


def load_execution_budget_receipt(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    receipt = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"invalid JSON number: {value}")))
    return validate_execution_budget_receipt(receipt), hashlib.sha256(raw).hexdigest()


class ParentResourceEvidenceError(RuntimeError):
    def __init__(self, reason: str, evidence: dict[str, Any]):
        self.evidence = evidence
        super().__init__(f"parent resource evidence rejected ({reason})")


def verify_parent_work_evidence(*, journal: Path, attempt_id: str, stop_path: Path,
                                stage: str, clock: Callable[[], float] = time.time,
                                reader: IncrementalRecordReader | None = None) -> dict[str, Any]:
    """Validate the latest complete parent sample at child use time; never sample here."""
    evidence: dict[str, Any] = {"stage": stage, "maximum_component_age_seconds": 2.0}
    if Path(stop_path).exists():
        evidence.update(validation_unix_time=float(clock()), reason="stop_requested",
            trigger="parent STOP_REQUESTED exists")
        raise ParentResourceEvidenceError("stop_requested", evidence)
    source: IncrementalRecordReader | None = reader
    try:
        source = reader or IncrementalRecordReader(Path(journal), attempt_id=attempt_id)
        envelope = source.refresh() if reader is not None else source.latest
    except Exception as exc:
        reason = "wrong_attempt" if "attempt id" in str(exc) else (
            "journal_integrity_or_read_failure" if "checksum" in str(exc) or "sequence" in str(exc)
            or "malformed" in str(exc) or "truncated" in str(exc)
            or isinstance(exc, json.JSONDecodeError) else "journal_read_failure")
        evidence.update(validation_unix_time=float(clock()), reason=reason, trigger=repr(exc)[:300])
        if source is not None and source.latest is not None:
            previous = source.latest
            previous_payload = previous.get("payload", {})
            evidence["last_verified_before_read_error"] = {
                "sequence": previous.get("sequence"), "sha256": previous.get("sha256"),
                "sample_id": previous_payload.get("sample_id"),
                "candidate_used": False}
        raise ParentResourceEvidenceError(reason, evidence) from exc
    if envelope is None:
        evidence["validation_unix_time"] = float(clock())
        evidence.update(reason="missing_sample", trigger="parent resource journal has no verified sample")
        raise ParentResourceEvidenceError("missing_sample", evidence)
    # Read and integrity validation are part of the handoff latency. Timestamp
    # the child decision after that work, so scanning never grants extra age.
    checked_at = float(clock())
    evidence["validation_unix_time"] = checked_at
    payload = dict(envelope.get("payload", {}))
    evidence.update(parent_attempt_id=envelope.get("attempt_id"),
        parent_sample_sequence=envelope.get("sequence"), parent_sample_sha256=envelope.get("sha256"),
        parent_sample_id=payload.get("sample_id"), parent_sample_stage=payload.get("stage"),
        gpu_read_completed_unix_time=payload.get("gpu_read_completed_unix_time"),
        host_read_completed_unix_time=payload.get("host_read_completed_unix_time"),
        gpu_measured_unix_time=payload.get("gpu_measured_unix_time"),
        host_sampled_unix_time=payload.get("host_sampled_unix_time"))
    def age(completed: Any) -> float | None:
        try:
            value = float(completed)
            return checked_at - value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None
    evidence["gpu_age_seconds"] = age(payload.get("gpu_read_completed_unix_time", payload.get("gpu_measured_unix_time")))
    evidence["host_age_seconds"] = age(payload.get("host_read_completed_unix_time", payload.get("host_sampled_unix_time")))
    if envelope.get("attempt_id") != attempt_id:
        reason = "wrong_attempt"
    elif payload.get("runtime_gate_status") != "passed":
        reason = "parent_sample_failed"
    else:
        try:
            failures = validate_ml_resource_sample(payload, launch=False,
                projected_remaining_write_bytes=0, wall_clock=lambda: checked_at)
        except Exception as exc:
            evidence.update(reason="malformed_sample", trigger=repr(exc)[:300])
            raise ParentResourceEvidenceError("malformed_sample", evidence) from exc
        evidence["validation_failures"] = failures[:12]
        if failures:
            freshness = [row for row in failures if row.get("metric", "").endswith("measurement_freshness")]
            if freshness and any(row.get("measurement_status") == "clock_discontinuity" for row in freshness):
                reason = "clock_discontinuity"
            elif freshness and any(row.get("measurement_status") == "stale" for row in freshness):
                reason = "stale_component"
            elif freshness:
                reason = "missing_or_malformed_component"
            else:
                reason = "parent_sample_failed"
        else:
            return {**evidence, "reason": None}
    evidence["reason"] = reason
    raise ParentResourceEvidenceError(reason, evidence)


def verify_accepted_handoff(output: Path, report_dir: Path, attempt_id: str) -> dict[str, Any]:
    """Accept only the complete child and parent success/ownership/hash chain."""
    output, report_dir = Path(output), Path(report_dir)
    for name in ("phase_failed.json", "STOP_REQUESTED"):
        if (output / name).exists() or (report_dir / "supervision" / name).exists():
            raise ValueError(f"failed or stopped pretraining attempt cannot be accepted: {name}")
    if (report_dir / "supervision" / "supervisor_error.json").exists():
        raise ValueError("parent supervisor error receipt blocks handoff acceptance")
    handoff = output / "phase_handoff.json"
    commit = output / "phase_handoff.commit"
    accepted = output / "phase_handoff.accepted.json"
    if not all(path.is_file() for path in (handoff, commit, accepted)):
        raise ValueError("pretraining handoff lacks child commit or final parent acceptance")
    digest = hashlib.sha256(handoff.read_bytes()).hexdigest()
    commit_row = json.loads(commit.read_text(encoding="utf-8"))
    if commit_row.get("phase_handoff_sha256") != digest or commit_row.get("status") != "prepared":
        raise ValueError("prepared child commit does not match handoff bytes")
    parent = json.loads(accepted.read_text(encoding="utf-8"))
    if (parent.get("status") != "accepted" or parent.get("attempt_id") != attempt_id
            or parent.get("phase_handoff_sha256") != digest):
        raise ValueError("final parent acceptance does not match handoff")
    journal = report_dir / "child_receipts.jsonl"
    rows = read_records(journal, attempt_id=attempt_id, allow_truncated_tail=False)
    if not rows or rows[-1]["payload"].get("kind") != "phase_complete":
        raise ValueError("child receipt chain has no terminal phase_complete record")
    child_done = rows[-1]["payload"]
    if (child_done.get("phase_handoff_sha256") != digest
            or parent.get("child_receipt_sha256") != rows[-1]["sha256"]):
        raise ValueError("child receipt and parent acceptance hash chain differs")
    if parent.get("owned_process_identities") is not None:
        started = next((row["payload"] for row in rows if row["payload"].get("kind") == "phase_started"), None)
        proof = (started or {}).get("ownership", {})
        parent_ids = {int(row["pid"]): row for row in parent["owned_process_identities"]}
        ancestry = proof.get("observed_ancestry", [])
        if len(ancestry) not in (1, 2) or int(ancestry[0].get("pid", -1)) not in parent_ids:
            raise ValueError("parent and child executing-process identities do not agree")
        if len(ancestry) == 2:
            launcher = ancestry[1]
            if (int(launcher.get("pid", -1)) not in parent_ids
                    or int(ancestry[0].get("ppid", -2)) != int(launcher["pid"])):
                raise ValueError("parent and child launcher identities do not agree")
    parent_attempt = parent.get("parent_attempt_id")
    if not isinstance(parent_attempt, str) or not parent_attempt:
        raise ValueError("parent acceptance does not identify its resource sample attempt")
    parent_dir = report_dir / "supervision"
    resource_rows = read_records(parent_dir / "runtime_resources.jsonl", attempt_id=parent_attempt,
        allow_truncated_tail=False)
    if (not resource_rows or resource_rows[-1]["sha256"] != parent.get("parent_final_sample_sha256")
            or resource_rows[-1]["payload"].get("stage") != "child_exit_finalization"
            or resource_rows[-1]["payload"].get("runtime_gate_status") != "passed"):
        raise ValueError("final parent resource sample is missing, failed, or hash-mismatched")
    owned = json.loads((parent_dir / "owned_process_stop_receipt.json").read_text(encoding="utf-8"))
    if (owned.get("status") != "child_exited" or int(owned.get("return_code", -1)) != 0
            or int(owned.get("pid", -1)) != int(parent.get("owned_pid", -2))
            or int(parent.get("owned_exit_code", -1)) != 0):
        raise ValueError("owned-child exit receipt does not prove a clean child exit")
    return parent


def _publish_parent_acceptance(path: Path, payload: dict[str, Any], *,
                               fsync_fn: Callable[[int], Any] = os.fsync,
                               replace_fn: Callable[[Path, Path], Any] = os.replace,
                               before_publish: Callable[[], Any] | None = None) -> None:
    """Publish the parent acceptance only after its complete temporary receipt is fsynced."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    data = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    with pending.open("xb") as stream:
        written = stream.write(data)
        if written != len(data): raise OSError("short parent acceptance write")
        stream.flush(); fsync_fn(stream.fileno())
    if before_publish is not None: before_publish()
    replace_fn(pending, path)


def _publish_immutable_json(path: Path, payload: dict[str, Any], *,
                           fsync_fn: Callable[[int], Any] = os.fsync,
                           replace_fn: Callable[[Path, Path], Any] = os.replace,
                           before_publish: Callable[[], Any] | None = None) -> None:
    path = Path(path); pending = path.with_name(path.name + ".pending")
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with pending.open("xb") as stream:
        if stream.write(data) != len(data): raise OSError("short immutable publication write")
        stream.flush(); fsync_fn(stream.fileno())
    if before_publish is not None: before_publish()
    replace_fn(pending, path)


def validate_launch(*, root: Path, output: Path, config: Path, cache: Path,
                    budget: dict[str, Any], execution_budget_path: Path,
                    execution_budget_sha256: str, protocol_sha256: str,
                    input_validator: Callable[..., Any] | None = None) -> dict[str, Any]:
    if Path(output).exists(): raise FileExistsError(f"fresh output is occupied: {output}")
    if Path(config).resolve() != (root / production.CONFIG_REL).resolve(): raise ValueError("foreign config path")
    if Path(cache).resolve() != (root / production.CACHE_REL).resolve(): raise ValueError("foreign cache path")
    current_budget, current_sha256 = load_execution_budget_receipt(execution_budget_path)
    if current_sha256 != execution_budget_sha256 or current_budget != budget:
        raise ValueError("execution budget changed after parent validation or differs from child envelope")
    protocol_path = Path(root) / "analysis/dqfd_pilot_worker_prompts_20261024/11_WIRE_PRETRAIN_PRODUCTION_CLI_PROMPT.txt"
    if PROMPT11_SHA256 != protocol_sha256:
        raise ValueError("protocol pin differs from preserved prompt11")
    if protocol_sha256 != PROMPT11_SHA256:
        raise ValueError("protocol SHA differs from the budget receipt's pinned original protocol")
    hashes = (input_validator or production._validate_inputs)(Path(root).resolve())
    return {"hashes": hashes, "protocol_sha256": protocol_sha256,
        "budget": dict(budget), "execution_budget_path": str(Path(execution_budget_path).resolve()),
        "execution_budget_sha256": current_sha256, "output": str(Path(output).resolve()),
        "root": str(Path(root).resolve()), "config": str(Path(config).resolve()),
        "cache": str(Path(cache).resolve()), "parent_pid": os.getpid()}


def private_child(*, envelope_path: Path, supplied_token: str, expected_attempt_id: str | None = None,
                  expected_execution_budget_path: Path | None = None,
                  expected_execution_budget_sha256: str | None = None,
                  dependencies: dict[str, Any] | None = None) -> int:
    """Private supervised child; dependency injection is programmatic only."""
    d = dependencies or {}; envelope_path = Path(envelope_path)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    if not hmac.compare_digest(str(envelope.get("token", "")), str(supplied_token)):
        raise PermissionError("private child parent token mismatch")
    if expected_attempt_id is not None and not hmac.compare_digest(
            str(envelope.get("attempt_id", "")), str(expected_attempt_id)):
        raise PermissionError("private child attempt identity mismatch")
    parent_pid = int(envelope.get("parent_pid", -1))
    if parent_pid <= 0: raise PermissionError("private child parent identity invalid")
    if d.get("parent_check") is not None and d.get("process_inspector") is None:
        # Compatibility for older in-process injected fixtures. Production
        # envelopes always carry process identities and never take this branch.
        if not d["parent_check"](parent_pid): raise PermissionError("private child parent identity rejected")
        ownership = {"status": "injected_legacy_fixture_only"}
    else:
        ownership = validate_private_child_ownership(child_pid=os.getpid(), parent_pid=parent_pid,
            parent_create_time=float(envelope.get("parent_create_time", -1)),
            parent_exe=str(envelope.get("parent_exe", "")),
            launcher_exe=str(envelope.get("launcher_exe", "")),
            child_exe=str(envelope.get("child_exe", "")),
            process_inspector=d.get("process_inspector", _process_record))
    launch = envelope["launch"]
    if (expected_execution_budget_path is not None
            and Path(expected_execution_budget_path).resolve() != Path(launch["execution_budget_path"]).resolve()):
        raise ValueError("private child execution budget path substitution rejected")
    if (expected_execution_budget_sha256 is not None
            and not hmac.compare_digest(str(expected_execution_budget_sha256),
                str(launch["execution_budget_sha256"]))):
        raise ValueError("private child execution budget hash substitution rejected")
    if launch.get("caps") != caps_for_budget(launch["budget"], EXPECTED_CAPS):
        raise ValueError("private child cap envelope differs from frozen limits")
    checked = validate_launch(root=Path(launch["root"]), output=Path(launch["output"]),
        config=Path(launch["config"]), cache=Path(launch["cache"]),
        budget=launch["budget"], execution_budget_path=Path(launch["execution_budget_path"]),
        execution_budget_sha256=launch["execution_budget_sha256"],
        protocol_sha256=launch["protocol_sha256"],
        input_validator=d.get("root_validator"))
    if checked["hashes"] != launch["hashes"]:
        raise ValueError("private child input identity changed after parent validation")
    code_hash = production_code_hash(Path(launch["root"]))
    if code_hash != launch.get("code_sha256"):
        raise ValueError("private child code hash differs from parent envelope")
    output = Path(launch["output"])
    journal = envelope_path.parent / "child_receipts.jsonl"; attempt = str(envelope["attempt_id"]); seq = 0
    if journal.exists(): raise FileExistsError(f"child receipt journal is occupied: {journal}")
    def receipt(row):
        nonlocal seq
        candidate = seq + 1
        try:
            append_record(journal, row, attempt_id=attempt, sequence=candidate)
        except BaseException:
            # A write/fsync error can occur after a complete checksummed line is
            # visible. Advance only when readback proves that record exists;
            # otherwise reuse the unconsumed sequence for phase_failed evidence.
            try:
                written = read_records(journal, attempt_id=attempt, allow_truncated_tail=True)
                if written: seq = int(written[-1]["sequence"])
            except Exception:
                pass
            raise
        seq = candidate
    learner = None
    status = "failed"
    child_clock = d.get("clock", time.monotonic)
    last_counters: dict[str, Any] = {key: None for key in
        ("attempted_steps", "applied_steps", "successful_steps", "failed_after_step",
         "unknown_step_outcomes", "target_copies")}
    last_update_evidence: dict[str, Any] | None = None
    last_resource_check_evidence: dict[str, Any] | None = None
    update_evidence_counts = {"applied_updates": 0, "target_copies": 0,
        "priority_updated_updates": 0, "diagnostic_receipted_updates": 0,
        "phase_successful_updates": 0}
    parent_resource_reader: IncrementalRecordReader | None = d.get("parent_resource_reader")
    last_episode: dict[str, Any] = {"attempted": None, "completed": None,
        "uncertain": None, "native_frames": None, "outcome": None}
    def guard(stage: str) -> dict[str, Any]:
        nonlocal parent_resource_reader, last_resource_check_evidence
        current_budget, current_sha256 = load_execution_budget_receipt(Path(launch["execution_budget_path"]))
        if current_sha256 != launch["execution_budget_sha256"] or current_budget != launch["budget"]:
            raise ValueError("execution budget changed after parent validation")
        elapsed = max(0.0, child_clock() - float(launch["supervised_started_at"]))
        check_original_total_deadline(elapsed, launch["budget"])
        hook = d.get("work_guard")
        if hook is not None:
            return dict(hook(stage))
        journal_path = launch.get("parent_resource_journal")
        if journal_path is None:
            # Programmatic pre-existing fixtures may omit a parent journal; the
            # connected CLI always carries it in the immutable parent envelope.
            return {"stage": stage, "parent_evidence": "injected-fixture-only"}
        try:
            if parent_resource_reader is None:
                parent_resource_reader = IncrementalRecordReader(Path(journal_path),
                    attempt_id=str(launch["parent_attempt_id"]))
                # Its initial full scan already validated the complete journal.
                evidence = verify_parent_work_evidence(journal=Path(journal_path),
                    attempt_id=str(launch["parent_attempt_id"]),
                    stop_path=Path(launch["parent_stop_path"]), stage=stage,
                    clock=d.get("wall_clock", time.time), reader=parent_resource_reader)
            else:
                evidence = verify_parent_work_evidence(journal=Path(journal_path),
                    attempt_id=str(launch["parent_attempt_id"]),
                    stop_path=Path(launch["parent_stop_path"]), stage=stage,
                    clock=d.get("wall_clock", time.time), reader=parent_resource_reader)
            last_resource_check_evidence = evidence
            return evidence
        except ParentResourceEvidenceError as exc:
            last_resource_check_evidence = dict(exc.evidence)
            try:
                receipt({"kind": "resource_check_failed", **exc.evidence,
                    "counters": _counter_snapshot(learner),
                    "update_evidence_counts": dict(update_evidence_counts),
                    "last_update_outcome": ({"successful_step": last_update_evidence.get("successful_step"),
                        "applied_steps": last_update_evidence.get("applied_steps"),
                        "target_copies": last_update_evidence.get("target_copies"),
                        "priority_updated": last_update_evidence.get("priority_updated"),
                        "diagnostic_receipted": last_update_evidence.get("diagnostic_receipted"),
                        "durability": last_update_evidence.get("durability")}
                        if last_update_evidence else None)})
            except Exception as receipt_error:
                exc.add_note(f"resource check evidence receipt failed: {receipt_error!r}")
            raise
    def close_learner():
        nonlocal learner
        if learner is None: return
        errors = []
        for resource in (learner, getattr(learner, "demo_index", None),
                getattr(getattr(learner, "demo_index", None), "cache", None)):
            close = getattr(resource, "close", None)
            if close:
                try: close()
                except Exception as exc: errors.append(repr(exc)[:400])
        learner = None
        if errors: raise RuntimeError("learner/demo reader cleanup failed: " + "; ".join(errors))
    try:
        guard("initialization")
        receipt({"kind": "phase_started", "status": "initializing", "parent_pid": parent_pid,
            "ownership": ownership,
            "input_hashes": checked["hashes"], "budget": launch["budget"]})
        factories = dict(d.get("factory_dependencies", {}))
        default_factory = production.create_fresh_demo_learner
        default_loop = run_demo_pretraining
        if launch["budget"].get("study_id") == "dqfd_demo_continuation_20261025":
            from .dqfd_continuation import create_continuation_learner, run_continuation
            default_factory, default_loop = create_continuation_learner, run_continuation
        learner = (d.get("learner_factory", default_factory))(
            root=Path(launch["root"]), device=d.get("device", "cuda:0"), **factories)
        learner.model_metadata.update({"protocol_sha256": launch["protocol_sha256"],
            "budget_sha256": launch["budget_sha256"], "code_sha256": launch["code_sha256"]})
        def update_receipt(row):
            nonlocal last_update_evidence
            last_counters.update(_counter_snapshot(learner))
            last_update_evidence = {"successful_step": row.get("successful_step"),
                "diagnostic": dict(row), "applied_steps": last_counters.get("applied_steps"),
                "target_copies": last_counters.get("target_copies"),
                "priority_updated": True, "diagnostic_receipted": False,
                "phase_success": False, "durability": "pending"}
            update_evidence_counts["applied_updates"] = int(last_counters.get("applied_steps") or 0)
            update_evidence_counts["target_copies"] = int(last_counters.get("target_copies") or 0)
            update_evidence_counts["priority_updated_updates"] += 1
            try:
                receipt({"kind": "update", **row,
                    "outcome_counts": {"applied_steps": last_counters.get("applied_steps"),
                        "target_copies": last_counters.get("target_copies"),
                        "priority_updated": True, "diagnostic_receipted": True,
                        "phase_success": False}})
                last_update_evidence["diagnostic_receipted"] = True
                last_update_evidence["durability"] = "fsynced"
                update_evidence_counts["diagnostic_receipted_updates"] += 1
            except BaseException:
                last_update_evidence["durability"] = "unknown_after_receipt_write_failure"
                raise
        def episode(update, checkpoint, label):
            guard("demo_encoding_enter")
            try:
                digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
                outcome = (d.get("episode_factory", production.production_reset389_episode))(
                    root=Path(launch["root"]), output_dir=output / "videos", receipt_path=journal,
                    attempt_id=attempt, update=update, label=label, checkpoint_path=Path(checkpoint),
                    checkpoint_sha256=digest, pins={"actor_sha256": checked["hashes"]["actor"],
                        "manifest_sha256": checked["hashes"]["manifest"], "READY_sha256": checked["hashes"]["READY"],
                        "identity_sha256": checked["hashes"]["identity"],
                        "protocol_sha256": launch["protocol_sha256"],
                        "budget_sha256": launch["budget_sha256"],
                        "code_sha256": launch["code_sha256"]}, receipt_fn=receipt,
                    **dict(d.get("episode_dependencies", {})))
                last_episode.update(attempted=outcome.get("native_step_attempted"),
                    completed=outcome.get("native_step_completed"), uncertain=outcome.get("native_step_uncertain"),
                    native_frames=outcome.get("native_frames"), outcome=outcome.get("end_reason"),
                    status=outcome.get("status"))
                guard("demo_encoding_leave")
                return outcome
            except BaseException as episode_error:
                # The evaluator writes an incomplete row before raising. Preserve
                # unknowns as null; never infer zero from absent reward/native data.
                try:
                    records = read_records(journal, attempt_id=attempt, allow_truncated_tail=True)
                    payload = records[-1]["payload"] if records else {}
                    if "native_step_attempted" in payload:
                        last_episode.update(attempted=payload.get("native_step_attempted"),
                            completed=payload.get("native_step_completed"), uncertain=payload.get("native_step_uncertain"),
                            native_frames=payload.get("native_frames"), outcome=payload.get("end_reason"),
                            status=payload.get("status"))
                except Exception as read_error:
                    episode_error.add_note(f"episode receipt readback also failed: {read_error!r}")
                raise
        result = (d.get("loop", default_loop))(learner, output_dir=output,
            episode_fn=episode, update_receipt_fn=update_receipt,
            checkpoint_receipt_fn=lambda row: receipt({"kind": "milestone_checkpoint", **row}),
            boundary_fn=guard,
            clock=child_clock, started_at=float(launch["supervised_started_at"]),
            execution_budget=launch["budget"])
        guard("finalization_enter")
        close_learner()
        total_supervised = max(0.0, child_clock() - float(launch["supervised_started_at"]))
        model_io_total = total_supervised - float(result["demo_gate_encoding_active_seconds"])
        check_original_total_deadline(total_supervised, launch["budget"])
        if (model_io_total > launch["budget"]["model_io_active_seconds"]
                or float(result["demo_gate_encoding_active_seconds"]) > launch["budget"]["demo_encoding_active_seconds"]):
            raise TimeoutError("complete supervised setup/update/demo/cleanup interval exceeded a frozen slice or cumulative cap")
        phase_handoff = {"status": "prepared", "fixed2000_retention_passed": bool(result["fixed2000_retention_passed"]),
            "successful_updates": int(result["successful_updates"]), "attempted_updates": int(result["attempted_updates"]),
            "target_copies": int(result["target_copies"]), "state_hashes": result["state_hashes"],
            "milestones": [{"update": row["update"], "checkpoint_sha256": row["checkpoint_sha256"],
                "terminal_valid": row.get("terminal_valid"), "end_reason": row.get("end_reason"),
                "video_sha256": row.get("video_sha256"), "video_bytes": row.get("video_bytes")}
                for row in result["milestones"]], "input_hashes": checked["hashes"],
            "protocol_sha256": launch["protocol_sha256"], "budget_sha256": launch["budget_sha256"],
            "code_sha256": launch["code_sha256"], "whole_supervised_active_seconds": total_supervised,
            "model_io_active_seconds_including_setup_cleanup": model_io_total,
            "demo_gate_encoding_active_seconds": result["demo_gate_encoding_active_seconds"]}
        handoff_path = output / "phase_handoff.json"
        _publish_immutable_json(handoff_path, phase_handoff,
            fsync_fn=d.get("publication_fsync", os.fsync),
            before_publish=lambda: guard("finalization_before_handoff_publish"))
        total_supervised = max(0.0, child_clock() - float(launch["supervised_started_at"]))
        model_io_total = total_supervised - float(result["demo_gate_encoding_active_seconds"])
        check_original_total_deadline(total_supervised, launch["budget"])
        if (model_io_total > launch["budget"]["model_io_active_seconds"]
                or float(result["demo_gate_encoding_active_seconds"]) > launch["budget"]["demo_encoding_active_seconds"]):
            raise TimeoutError("complete supervised setup/update/demo/cleanup interval exceeded a frozen slice or cumulative cap")
        handoff_digest = hashlib.sha256(handoff_path.read_bytes()).hexdigest()
        commit_path = output / "phase_handoff.commit"
        retention_passed = bool(result["fixed2000_retention_passed"])
        completion_status = "complete" if retention_passed else "pretraining_complete_retention_failed"
        _publish_immutable_json(commit_path, {"status": "prepared",
                "phase_handoff_sha256": handoff_digest, "fixed2000_retention_passed": retention_passed},
            fsync_fn=d.get("publication_fsync", os.fsync),
            before_publish=lambda: guard("finalization_before_commit_publish"))
        guard("finalization_after_commit_fsync")
        total_supervised = max(0.0, child_clock() - float(launch["supervised_started_at"]))
        model_io_total = total_supervised - float(result["demo_gate_encoding_active_seconds"])
        check_original_total_deadline(total_supervised, launch["budget"])
        if (model_io_total > launch["budget"]["model_io_active_seconds"]
                or float(result["demo_gate_encoding_active_seconds"]) > launch["budget"]["demo_encoding_active_seconds"]):
            raise TimeoutError("complete supervised setup/update/demo/cleanup interval exceeded a frozen slice or cumulative cap")
        completed_update_counts = {**update_evidence_counts,
            "phase_successful_updates": int(result["successful_updates"])}
        receipt({"kind": "phase_complete", "status": completion_status,
            "retention_gate_status": "passed" if retention_passed else "failed",
            "successful_updates": result["successful_updates"],
            "update_evidence_counts": completed_update_counts,
            "fixed2000_retention_passed": retention_passed,
            "state_hashes": result["state_hashes"], "phase_handoff_sha256": handoff_digest,
            "whole_supervised_active_seconds": total_supervised,
            "model_io_active_seconds_including_setup_cleanup": model_io_total,
            "demo_gate_encoding_active_seconds": result["demo_gate_encoding_active_seconds"]})
        status = "complete"
        update_evidence_counts.update(completed_update_counts)
        return 0
    except BaseException as exc:
        last_counters.update(_counter_snapshot(learner))
        try: close_learner()
        except Exception as close_error: exc.add_note(f"learner cleanup also failed: {close_error!r}")
        try:
            if output.exists():
                failure_path = output / "phase_failed.json"
                if not failure_path.exists():
                    with failure_path.open("xb") as stream:
                        stream.write((json.dumps({"status": "failed", "error": repr(exc)[:1000],
                            "secondary_errors": list(getattr(exc, "__notes__", ()))[:8],
                            "counters": last_counters, "last_episode": last_episode,
                            "last_update_outcome": last_update_evidence,
                            "last_resource_check": last_resource_check_evidence,
                            "update_evidence_counts": update_evidence_counts,
                            "outcome_known": last_episode.get("outcome") is not None}, sort_keys=True) + "\n").encode())
                        stream.flush(); os.fsync(stream.fileno())
        except Exception as failure_file_error:
            exc.add_note(f"phase failure file write failed: {failure_file_error!r}")
        try:
            receipt({"kind": "phase_failed", "status": "failed", "error": repr(exc)[:1000],
                "secondary_errors": list(getattr(exc, "__notes__", ()))[:8],
                "counters": last_counters, "last_episode": last_episode,
                "last_update_outcome": last_update_evidence,
                "last_resource_check": last_resource_check_evidence,
                "update_evidence_counts": update_evidence_counts,
                "outcome_known": last_episode.get("outcome") is not None})
        except Exception as receipt_error:
            exc.add_note(f"failure receipt failed: {receipt_error!r}")
        raise
    finally:
        if learner is not None: close_learner()


def dispatch_pretrain(*, root: Path, output: Path, config: Path, cache: Path,
                      protocol_sha256: str, execution_budget_path: Path, dry_run: bool,
                      report_dir: Path, dependencies: dict[str, Any] | None = None) -> dict[str, Any]:
    d = dependencies or {}
    parent_clock = d.get("monotonic_clock", time.monotonic)
    supervised_started_at = parent_clock()
    budget, sidecar_sha256 = load_execution_budget_receipt(execution_budget_path)
    launch = validate_launch(root=root, output=output, config=config, cache=cache,
        budget=budget, execution_budget_path=execution_budget_path,
        execution_budget_sha256=sidecar_sha256, protocol_sha256=protocol_sha256,
        input_validator=d.get("root_validator"))
    launch["caps"] = caps_for_budget(budget, EXPECTED_CAPS)
    launch["supervised_started_at"] = supervised_started_at
    launch["budget_sha256"] = hashlib.sha256(json.dumps(budget, sort_keys=True).encode()).hexdigest()
    launch["code_sha256"] = production_code_hash(Path(root))
    token = secrets.token_hex(32); attempt = uuid.uuid4().hex
    supervisor_report_dir = Path(d.get("supervisor_report_dir", report_dir / "supervision"))
    supervisor_attempt = uuid.uuid4().hex
    launch["parent_attempt_id"] = supervisor_attempt
    launch["parent_resource_journal"] = str(supervisor_report_dir / "runtime_resources.jsonl")
    launch["parent_stop_path"] = str(supervisor_report_dir / "STOP_REQUESTED")
    parent_identity = _process_record(os.getpid())
    envelope = {"parent_pid": os.getpid(), "parent_create_time": parent_identity["create_time"],
        "parent_exe": parent_identity["exe"], "launcher_exe": os.path.abspath(sys.executable),
        "child_exe": os.path.abspath(getattr(sys, "_base_executable", sys.executable)),
        "token": token, "attempt_id": attempt, "launch": launch}
    report_dir = Path(report_dir)
    if report_dir.exists(): raise FileExistsError(f"fresh entrypoint report is occupied: {report_dir}")
    report_dir.mkdir(parents=True)
    envelope_path = report_dir / "private_child_envelope.json"
    envelope_path.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    command = [sys.executable, str(Path(root) / "run.py"), "level1_dqfd_pilot", "pretrain",
        "--_private-child", "--envelope", str(envelope_path), "--token", token,
        "--_attempt-id", attempt, "--_execution-budget", launch["execution_budget_path"],
        "--_execution-budget-sha256", launch["execution_budget_sha256"]]
    if dry_run:
        envelope["token"] = "[REDACTED_DRY_RUN_TOKEN]"
        envelope_path.write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
        command[command.index("--token") + 1] = "[REDACTED_DRY_RUN_TOKEN]"
        return {"status": "validated_dry_run_no_child", "child_argv": command,
            "launch": launch, "parent_token_generated": True, "optimizer_or_native_initialized": False}
    supervisor = d.get("supervisor", supervise_replay)
    supervisor_kwargs = dict(cwd=Path(root), report_dir=Path(d.get("supervisor_report_dir", report_dir / "supervision")),
        output=Path(output), frame_cap=launch["caps"]["whole_pilot_native_frames_remaining"],
        artifact_cap=launch["caps"]["retained_bytes_total_max"],
        wall_cap=float(budget["remaining_seconds_deadline"]),
        slice_cap=float(budget["slice_active_seconds"]), require_gpu=True,
        sample_period=RESOURCE_SAMPLE_PERIOD_SECONDS, started_at=supervised_started_at, attempt_id=supervisor_attempt,
        child_receipt_path=report_dir / "child_receipts.jsonl", child_attempt_id=attempt,
        expected_child_exe=os.path.abspath(getattr(sys, "_base_executable", sys.executable)))
    for name, key, default in (("sample_fn", "sample_fn", None), ("gpu_sampler", "gpu_sampler", None),
            ("clock", "supervisor_clock", parent_clock), ("sleeper", "sleeper", time.sleep),
            ("popen_factory", "popen_factory", None)):
        value = d.get(key, default)
        if value is not None: supervisor_kwargs[name] = value
    result = supervisor(command, **supervisor_kwargs)
    if result.get("status") == "child_exited" and int(result.get("return_code", -1)) == 0:
        supervised_report = Path(supervisor_kwargs["report_dir"])
        resource_rows = read_records(supervised_report / "runtime_resources.jsonl",
            attempt_id=supervisor_attempt, allow_truncated_tail=False)
        final_sample = resource_rows[-1]["payload"] if resource_rows else {}
        owned = json.loads((supervised_report / "owned_process_stop_receipt.json").read_text(encoding="utf-8"))
        if (final_sample.get("stage") != "child_exit_finalization"
                or final_sample.get("runtime_gate_status") != "passed"
                or owned.get("status") != "child_exited" or int(owned.get("return_code", -1)) != 0
                or int(owned.get("pid", -1)) != int(result.get("pid", -2))):
            raise RuntimeError("parent final sample/owned-child receipt failed acceptance: " +
                repr({"stage":final_sample.get("stage"), "gate":final_sample.get("runtime_gate_status"),
                    "owned_status":owned.get("status"), "owned_code":owned.get("return_code"),
                    "owned_pid":owned.get("pid"), "result_pid":result.get("pid")}))
        child_rows = read_records(report_dir / "child_receipts.jsonl",
            attempt_id=attempt, allow_truncated_tail=False)
        if not child_rows or child_rows[-1]["payload"].get("kind") != "phase_complete":
            raise RuntimeError("child did not finish its immutable receipt chain")
        handoff_hash = hashlib.sha256((Path(output) / "phase_handoff.json").read_bytes()).hexdigest()
        accepted = {"status": "accepted", "attempt_id": attempt,
            "phase_handoff_sha256": handoff_hash, "child_receipt_sha256": child_rows[-1]["sha256"],
            "parent_attempt_id": supervisor_attempt,
            "parent_final_sample_sha256": resource_rows[-1]["sha256"],
            "owned_pid": int(result["pid"]), "owned_exit_code": 0,
            "owned_process_identities": result.get("owned_process_identities")}
        accepted_path = Path(output) / "phase_handoff.accepted.json"
        try:
            def parent_deadline_check():
                current_budget, current_sha256 = load_execution_budget_receipt(
                    Path(launch["execution_budget_path"]))
                if current_sha256 != launch["execution_budget_sha256"] or current_budget != budget:
                    raise ValueError("execution budget changed after parent validation")
                elapsed_parent = parent_clock() - supervised_started_at
                check_original_total_deadline(elapsed_parent, budget)
                if (elapsed_parent >= float(budget["slice_active_seconds"])
                        or elapsed_parent >= float(budget["remaining_seconds_deadline"])):
                    raise TimeoutError("parent acceptance publication crossed the frozen slice deadline")
            _publish_parent_acceptance(accepted_path, accepted, before_publish=parent_deadline_check)
        except BaseException as exc:
            try:
                failure_path = Path(output) / "phase_failed.json"
                with failure_path.open("xb") as stream:
                    stream.write((json.dumps({"status": "failed", "boundary": "parent_acceptance",
                        "error": repr(exc)[:1000]}, sort_keys=True) + "\n").encode())
                    stream.flush(); os.fsync(stream.fileno())
            except Exception as secondary:
                exc.add_note(f"parent acceptance failure receipt also failed: {secondary!r}")
            raise
        verify_accepted_handoff(Path(output), report_dir, attempt)
    return {"status": result.get("status", "child_exited"), "child_argv": command,
        "launch": launch, "supervisor": result}
