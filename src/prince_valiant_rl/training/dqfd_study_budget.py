"""Explicit second-study budget amendment; legacy receipts stay frozen.

This module is data-only. It never imports Torch or probes hardware. The user
authorized more time and reuse of the verified demonstrations on 2026-10-25.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[3]
STUDY_ID = "dqfd_demonstration_extension_20261025"
SCHEMA = "dqfd_extended_execution_budget_v2"
AUTHORIZATION_REL = "config/dqfd_demonstration_extension_20261025.json"
PREDECESSOR_REL = "analysis/level1_dqfd_pretraining_live_v8_20261025/budget_reconciliation.json"
TICKS = 10_000_000
PROTOCOL = MappingProxyType({"original_total_active_seconds": 12600,
    "model_io_active_seconds": 1800, "demo_encoding_active_seconds": 1800,
    "slice_active_seconds": 3600, "updates": 2000, "batch_size": 128,
    "beta": .6, "seed": 20261024})
ALLOCATION = MappingProxyType({"additional_active_seconds": 7200,
    "effective_total_active_seconds": 12600, "model_io_active_seconds": 1800,
    "demo_encoding_active_seconds": 1800, "slice_active_seconds": 3600,
    "fresh_optimizer_updates": 2000, "cumulative_optimizer_step_ceiling": 4160,
    "historical_optimizer_steps": 2107, "additional_native_frames": 108000,
    "whole_native_frame_ceiling": 792640, "historical_native_frames": 81014,
    "slice_episode_count_max": 3, "slice_native_frames_max": 108000,
    "retained_cap_bytes": 12884901888})
FIELDS = frozenset({"schema", "study_id", "authorization_sha256", "predecessor_sha256",
    "protocol_sha256", "original_total_ticks", "prior_charged_ticks", "remaining_ticks",
    "preparation_charge_ticks", "accounting_basis", "receipt_sha256"})


def canonical_hash(row):
    return hashlib.sha256(json.dumps({k:v for k,v in row.items() if k != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read_bound(relative, expected_hash):
    path = ROOT / relative
    if path.stat().st_size > 65536:
        raise ValueError("study linkage file exceeds bounded JSON size")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("study authorization/predecessor hash mismatch")
    return json.loads(raw.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_extended_receipt(receipt, *, protocol_sha256):
    if set(receipt) != FIELDS or receipt.get("schema") != SCHEMA or receipt.get("study_id") != STUDY_ID:
        raise ValueError("extended study budget schema/fields differ")
    if receipt.get("receipt_sha256") != canonical_hash(receipt):
        raise ValueError("extended study receipt hash mismatch")
    if receipt.get("protocol_sha256") != protocol_sha256:
        raise ValueError("extended study learning protocol mismatch")
    authorization = _read_bound(AUTHORIZATION_REL, receipt["authorization_sha256"])
    if (authorization.get("study_id") != STUDY_ID or authorization.get("allocation") != dict(ALLOCATION)
            or authorization.get("training_initialization") != "fresh; no failed-phase resume"
            or not isinstance(authorization.get("user_authorization"), str)
            or not authorization["user_authorization"].strip()):
        raise ValueError("extended study authorization/allocation differs")
    predecessor = _read_bound(PREDECESSOR_REL, receipt["predecessor_sha256"])
    if (authorization.get("predecessor_sha256") != receipt["predecessor_sha256"]
            or predecessor.get("original_total_ticks") != 5400 * TICKS
            or predecessor.get("final_remaining_after_finalization_residual_ticks") != 21_923_809_324):
        raise ValueError("extended study predecessor does not preserve final through-freeze ledger")
    names = ("original_total_ticks", "prior_charged_ticks", "remaining_ticks", "preparation_charge_ticks")
    if any(type(receipt.get(name)) is not int or receipt[name] < 0 for name in names):
        raise ValueError("extended study ticks must be nonnegative integers")
    total, prior, remaining, preparation = (receipt[name] for name in names)
    historical = 5400 * TICKS - 21_923_809_324
    if (total != PROTOCOL["original_total_active_seconds"] * TICKS
            or prior != historical + preparation or prior >= total or remaining != total-prior):
        raise ValueError("extended study cumulative accounting does not reconcile")
    if not isinstance(receipt.get("accounting_basis"), str) or not receipt["accounting_basis"].strip():
        raise ValueError("extended study accounting basis missing")
    values = dict(PROTOCOL)
    values.update(prior_charged_active_seconds=f"{prior/TICKS:.7f}",
        remaining_original_active_seconds=f"{remaining/TICKS:.7f}",
        remaining_seconds_deadline=math.nextafter(remaining/TICKS, -math.inf),
        prior_charged_ticks=prior, remaining_active_ticks=remaining,
        slice_active_ticks=PROTOCOL["slice_active_seconds"]*TICKS,
        model_io_active_ticks=PROTOCOL["model_io_active_seconds"]*TICKS,
        demo_encoding_active_ticks=PROTOCOL["demo_encoding_active_seconds"]*TICKS,
        base_protocol_sha256=hashlib.sha256(json.dumps(dict(PROTOCOL),sort_keys=True,separators=(",", ":")).encode()).hexdigest(),
        execution_budget_receipt_sha256=receipt["receipt_sha256"],
        study_id=STUDY_ID, authorization_sha256=receipt["authorization_sha256"],
        legacy_original_total_active_seconds=5400,
        whole_pilot_native_frames_remaining=ALLOCATION["whole_native_frame_ceiling"]-ALLOCATION["historical_native_frames"])
    return values


def validate_loop_limits(budget):
    if budget.get("study_id") == STUDY_ID:
        expected = PROTOCOL
    elif "study_id" not in budget:
        expected = {"original_total_active_seconds": 5400, "updates": 2000,
            "model_io_active_seconds":600, "demo_encoding_active_seconds":900,
            "slice_active_seconds":1500}
    else:
        raise ValueError("unknown pretraining study")
    if any(budget.get(key) != value for key,value in expected.items()):
        raise ValueError("pretraining loop requires the validated frozen or explicitly amended execution budget")


def caps_for_budget(budget, legacy_caps):
    caps = dict(legacy_caps)
    if budget.get("study_id") == "dqfd_demo_continuation_20261025":
        caps["whole_pilot_native_frames_remaining"] = 747626
    if budget.get("study_id") == STUDY_ID:
        caps["whole_pilot_native_frames_remaining"] = ALLOCATION["whole_native_frame_ceiling"]-ALLOCATION["historical_native_frames"]
    return caps
