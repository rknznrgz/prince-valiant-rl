"""Preflight or explicitly invoke a configured source-action transition adapter."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from prince_valiant_rl.training.dqfd_transitions import (
    ACTION_MASKS, ROW_COUNT, SOURCES, V3_CACHE_CAP_BYTES, TransitionError, _atomic_json,
    build_cache, frame_index_capacity, inspect_npz,
)
from prince_valiant_rl.training.dqfd_supervision import supervise_replay
from prince_valiant_rl.training.dqfd_receipts import append_record

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "training/level1_full_route_recovery_replay_20261023/training_run/joint_training_data.npz"
CONFIG = ROOT / "training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json"
ROUTE = ROOT / "training/backward_level1_v1/route.npz"
ROM = ROOT / "roms/prince_valiant.nes"
STAGE1 = ROOT / "analysis/level1_dqfd_pilot_preparation_20261024"
EXPECTED = {
    "archive": "2cc2fd52ae5eb23d042368f33c108d58d57b539777addd6bb33eedbba63f6103",
    "config": "f2a28db6efc046b8527fab61b39a21fff2de8e3762eedc4638f0eb8e193f6d7d",
    "route": "214b5d56f858a45690779c11441ee3251880a1e8bf3418e564f461fe655b6eed",
}


def _write_parent_failure_receipts(report_dir: Path, lifecycle: dict, payload: dict) -> tuple[Path, Path]:
    """Keep a worker's earlier failure receipt intact when its parent reports exit."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stage_path = report_dir / ("parent_stage.json" if (report_dir / "stage.json").exists() else "stage.json")
    error_path = report_dir / ("parent_error.json" if (report_dir / "error.json").exists() else "error.json")
    for path, value in ((stage_path, {"status": "error", **lifecycle}), (error_path, payload)):
        with path.open("xb") as stream:
            stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    return stage_path, error_path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tree_bytes(path: Path) -> int:
    root = Path(path)
    if not root.is_dir():
        raise TransitionError(f"preserved artifact root is missing: {root}")
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def verify_preserved_budget() -> dict:
    stage1 = ROOT / "analysis/level1_dqfd_pilot_preparation_20261024"
    stage2 = ROOT / "analysis/level1_dqfd_transition_builder_implementation_20261024"
    stage3 = ROOT / "analysis/level1_dqfd_reconstruction_20261024"
    stage4 = ROOT / "analysis/level1_dqfd_reconstruction_v2_20261024"
    stage3_result = json.loads((stage3 / "result.json").read_text(encoding="utf-8"))
    stage4_result = json.loads((stage4 / "result.json").read_text(encoding="utf-8"))
    stage4_budget = json.loads((stage4 / "budget_amendment.json").read_text(encoding="utf-8"))
    stage3_cache = tree_bytes(ROOT / "training/level1_dqfd_transitions_reconstruction_20261024")
    stage4_cache = tree_bytes(ROOT / "training/level1_dqfd_transitions_reconstruction_v2_20261024")
    stage5_cache = tree_bytes(ROOT / "training/level1_dqfd_transitions_reconstruction_v3_20261024")
    stage5 = ROOT / "analysis/level1_dqfd_reconstruction_v3_20261024"
    report_sizes = {"stage1": tree_bytes(stage1), "stage2": tree_bytes(stage2),
                    "stage3": tree_bytes(stage3), "stage4": tree_bytes(stage4),
                    "stage5": tree_bytes(stage5)}
    stage1_to3_report = sum(report_sizes[k] for k in ("stage1", "stage2", "stage3"))
    preserved = stage3_cache + stage4_cache + stage1_to3_report + report_sizes["stage4"] + stage5_cache + report_sizes["stage5"]
    if (stage3_cache != int(stage3_result["cache"]["bytes"])
            or stage4_cache != int(stage4_result["cache"]["logical_file_bytes"])
            or stage1_to3_report != int(stage4_result["artifact_accounting"]["prior_stage1_to_stage3_report_bytes"])
            or report_sizes["stage4"] != int(stage4_budget["actual_attempt"]["report_bytes"])
            or stage5_cache != int(json.loads((stage5 / "result.json").read_text(encoding="utf-8"))["artifact_accounting"]["v3_partial_cache_logical_bytes"])
            or preserved != 7337261370):
        raise TransitionError("preserved partial cache/report bytes do not reconcile to frozen receipts")
    frames = int(stage3_result["native_frames"]) + int(stage4_result["native_frames"])
    seconds = float(stage3_result["timing_and_budget"]["active_reconstruction_wall_seconds"])
    seconds += float(stage4_result["active_wall_seconds_including_parent_preflight"])
    stage5_result = json.loads((stage5 / "result.json").read_text(encoding="utf-8"))
    frames += int(stage5_result["native_frames"]["total"])
    seconds += float(stage5_result["active_wall_seconds"]["parent_supervised_phase"])
    if frames != 23394 or abs(seconds - 244.061501) > 1e-6:
        raise TransitionError("prior native-frame or active-wall charges do not reconcile")
    return {"stage3_cache_bytes": stage3_cache, "stage4_cache_bytes": stage4_cache,
        "report_bytes": report_sizes, "stage1_to_stage3_report_bytes": stage1_to3_report,
        "stage5_partial_cache_bytes": stage5_cache, "stage5_report_bytes": report_sizes["stage5"],
        "preserved_logical_bytes": preserved, "prior_native_frames": frames,
        "prior_active_seconds": seconds, "native_frames_remaining": 589246,
        "active_seconds_remaining": 5155.938499}


def bounded_array(zf: zipfile.ZipFile, key: str) -> np.ndarray:
    # These lineage/action arrays have at most tens of thousands of scalar
    # values. Never use this helper for observation members.
    with zf.open(key + ".npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        elif version in ((2, 0), (3, 0)):
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise TransitionError(f"unsupported NPY header version {version}: {key}")
        size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if size > 8 * 1024 * 1024:
            raise TransitionError(f"preflight metadata member is unexpectedly large: {key} ({size} bytes)")
        raw = f.read(size)
        if len(raw) != size:
            raise TransitionError(f"truncated bounded metadata member: {key}")
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()


def preflight(args: argparse.Namespace) -> dict:
    if args.frame_cap != 76606 or args.wall_cap_seconds != 1255:
        raise TransitionError("v4 reconstruction requires exactly 76606 frames and 1255 live seconds")
    if args.artifact_cap_bytes != V3_CACHE_CAP_BYTES:
        raise TransitionError("v4 reconstruction artifact cap must be exactly 1073741824 bytes")
    paths = {"archive": ROOT / args.source, "config": ROOT / args.config, "route": ROUTE}
    actual = {key: digest(value) for key, value in paths.items()}
    for key, expected in EXPECTED.items():
        if actual[key] != expected:
            raise TransitionError(f"pinned {key} SHA-256 changed: expected={expected} actual={actual[key]}")
    protocol = json.loads((STAGE1 / "protocol.json").read_text(encoding="utf-8"))
    protocol_hash = digest(STAGE1 / "protocol.json")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    contract = protocol["environment_contract"]
    for key, expected in (("action_masks", list(ACTION_MASKS)), ("frame_skip", 1), ("gamma", 0.99997),
                          ("feature_schema_version", "level1_backward_v5_rea_gaps"),
                          ("observation_version", "rea_expanded_v2"),
                          ("history_offsets", [-60, -30, -15, -9, -6, -3, 0]),
                          ("reward_version", "level1_phase_corridor_v6"),
                          ("potential_version", "level1_type14_potential_v1"),
                          ("score_decrease_policy", "quarantine_valid_bcd_unexplained_decrease")):
        if config.get(key) != expected:
            raise TransitionError(f"pinned config contract mismatch for {key}")
    if config.get("weights") != contract.get("weights"):
        raise TransitionError("pinned reward weights differ from frozen protocol")
    amendment_path = ROOT / args.protocol_amendment
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    attempt = amendment.get("requested_attempt", {})
    if (attempt.get("native_frame_cap") != 76606 or attempt.get("wall_seconds_cap") != 1255 or
            attempt.get("cache_bytes_cap") != V3_CACHE_CAP_BYTES or
            amendment.get("cumulative_remaining_before_attempt", {}).get("native_frames") != 589246 or
            amendment.get("cumulative_remaining_before_attempt", {}).get("active_seconds") != 5155.938499):
        raise TransitionError("v4 accounting/budget amendment does not preserve the deducted remaining budget")
    if amendment.get("original_protocol_sha256") != protocol_hash:
        raise TransitionError("protocol amendment does not pin the current frozen stage-1 protocol")
    provenance_path = ROOT / "training/level1_full_route_recovery_replay_20261023/training_run/data_provenance.json"
    route_metadata_path = ROOT / "training/backward_level1_v1/route.json"
    lineage_hashes = {"data_provenance": digest(provenance_path), "route_metadata": digest(route_metadata_path)}
    if lineage_hashes["data_provenance"] != amendment["pinned_inputs"]["data_provenance_sha256"]:
        raise TransitionError("source data provenance hash changed")
    if lineage_hashes["route_metadata"] != amendment["pinned_inputs"]["route_metadata_sha256"]:
        raise TransitionError("route metadata hash changed")
    lineage_hashes["rom"] = digest(ROM)
    if lineage_hashes["rom"] != amendment["pinned_inputs"]["rom_sha256"]:
        raise TransitionError("ROM hash changed")
    route_meta = json.loads(route_metadata_path.read_text(encoding="utf-8"))
    reset_plan = []
    route_states = sorted(map(int, route_meta["states"]))
    for name, count, reset_frame, prefix, _kind in SOURCES:
        base = max((frame for frame in route_states if frame <= reset_frame), default=None)
        if base is None:
            raise TransitionError(f"no verified route checkpoint precedes reset {reset_frame}")
        checkpoint_name = route_meta["states"][str(base)]
        checkpoint_path = ROOT / "training/backward_level1_v1/states" / checkpoint_name
        checkpoint_hash = digest(checkpoint_path)
        pinned = amendment["pinned_inputs"]["route_reset_states"].get(str(reset_frame))
        if base != reset_frame or not pinned or pinned.get("path") != checkpoint_path.relative_to(ROOT).as_posix() or pinned.get("sha256") != checkpoint_hash:
            raise TransitionError(f"reset checkpoint lineage mismatch at route frame {reset_frame}")
        reset_plan.append({"source": name, "reset_frame": reset_frame, "snapshot_path": pinned["path"],
                           "snapshot_sha256": checkpoint_hash, "snapshot_load_native_frames": 0,
                           "fallback_route_replay_frames": reset_frame - base})
    if protocol["environment_contract"]["reference_config_sha256"] != EXPECTED["config"]:
        raise TransitionError("stage-1 protocol config hash differs from pinned config")
    details = inspect_npz(paths["archive"])
    headers = details["headers"]
    required = {"trajectory_id", "frame", "action", "action_mask", "provenance_kind", "post_ram_sha256"}
    if not required.issubset(headers):
        raise TransitionError(f"archive missing source metadata members: {sorted(required - set(headers))}")
    with zipfile.ZipFile(paths["archive"]) as zf:
        arrays = {k: bounded_array(zf, k) for k in required}
    ids, frames, actions = arrays["trajectory_id"], arrays["frame"], arrays["action"]
    masks, kinds = arrays["action_mask"], arrays["provenance_kind"]
    if len(ids) != ROW_COUNT or any(len(x) != ROW_COUNT for x in arrays.values()):
        raise TransitionError("archive row totals do not match pinned whitelist")
    cursor, source_plan = 0, []
    for trajectory_id, (name, count, reset, prefix, kind) in enumerate(SOURCES):
        end = cursor + count
        if not np.all(ids[cursor:end] == trajectory_id):
            raise TransitionError(f"source partition identity/order mismatch: {name}")
        expected_frames = np.arange(reset + prefix, reset + prefix + count, dtype=frames.dtype)
        if not np.array_equal(frames[cursor:end], expected_frames):
            raise TransitionError(f"source frame range mismatch: {name}")
        if not np.all(kinds[cursor:end] == kind):
            raise TransitionError(f"source provenance kind mismatch: {name}")
        if np.any(actions[cursor:end] < 0) or np.any(actions[cursor:end] >= len(ACTION_MASKS)):
            raise TransitionError(f"unsupported action index in {name}")
        if not np.array_equal(np.asarray(ACTION_MASKS, dtype=masks.dtype)[actions[cursor:end]], masks[cursor:end]):
            raise TransitionError(f"action-index/raw-mask mapping mismatch: {name}")
        source_plan.append({"name": name, "rows": count, "reset_frame": reset,
                            "context_prefix_frames": prefix, "source_kind": kind,
                            "first_frame": int(frames[cursor]), "last_frame": int(frames[end - 1])})
        cursor = end
    storage = sum(x["nbytes"] for x in headers.values())
    observation_keys = set(headers) - {"frame", "action", "action_mask", "confidence", "changed_action",
                                      "previous_route_mask", "post_ram_sha256", "provenance_kind", "trajectory_id"}
    observation_bytes = sum(int(headers[key]["nbytes"]) for key in observation_keys)
    storage_plan = frame_index_capacity(headers, offsets=config["history_offsets"])
    projected_cache_bytes = int(storage_plan["projected_total_bytes"])
    if not storage_plan["fits"] or projected_cache_bytes > args.artifact_cap_bytes:
        raise TransitionError(f"projected frame-indexed cache {projected_cache_bytes} bytes exceeds 1 GiB")
    online_frames = 32768 + 6 * (60 + 10 + 1)
    online_frame_bytes = online_frames * int(storage_plan["bytes_per_encoded_frame"])
    online_ref_generation_bytes = (32768 * 3 * 7 + 6 * 11 * 7) * 16
    online_index_bytes = online_frames * 64
    online_metadata_and_per_bytes = 100 * 1024**2
    online_staging_bytes = 128 * 1024**2
    online_projection = (online_frame_bytes + online_ref_generation_bytes + online_index_bytes
                         + online_metadata_and_per_bytes + online_staging_bytes)
    if online_projection > 2 * 1024**3:
        raise TransitionError("future frame-indexed online store exceeds its 2 GiB reserved capacity")
    preserved = verify_preserved_budget()
    report_reserve = 50 * 1024**2
    whole_pilot_ceiling = (preserved["preserved_logical_bytes"] + args.artifact_cap_bytes
        + 2 * 1024**3 + 1024**3 + report_reserve)
    if whole_pilot_ceiling > 12 * 1024**3:
        raise TransitionError("remaining pilot storage ledger exceeds its 12 GiB cumulative cap")
    return {"status": "preflight_passed_nonexecuting", "data_ready": False,
            "offline_builder_checks_passed": False, "transition_parity_passed": False,
            "trainer_ready": False, "source_rows": ROW_COUNT, "context_frames": 21,
            "sources": source_plan, "archive_headers": headers,
            "source_input_uncompressed_bytes": storage,
            "observation_input_uncompressed_bytes": observation_bytes,
            "frame_index_capacity_from_headers": storage_plan,
            "projected_cache_bytes_frame_indexed_v2": projected_cache_bytes,
            "preserved_artifact_ledger": preserved,
            "future_online_store_projection": {"capacity_transitions": 32768,
                "max_live_frame_count": online_frames, "frame_array_bytes": online_frame_bytes,
                "conservative_reference_and_generation_bytes": online_ref_generation_bytes,
                "frame_identity_index_bytes": online_index_bytes,
                "PER_owner_episode_and_metadata_reserve_bytes": online_metadata_and_per_bytes,
                "temporary_and_staging_bytes": online_staging_bytes,
                "projected_total_bytes": online_projection,
                "reserved_bytes": 2 * 1024**3,
                "live_retention_contract": "Keep each referenced native frame and generation alive while any resident start, one-step successor, paired n-step endpoint, terminal post-state, or owner-local queue references it; owner reset increments generation and flushes its queue; eviction only after all references retire."},
            "whole_remaining_pilot_storage_ledger": {"preserved_artifacts_bytes": preserved["preserved_logical_bytes"],
                "new_cache_cap_bytes": args.artifact_cap_bytes,
                "future_online_store_reserve_bytes": 2 * 1024**3,
                "future_model_optimizer_checkpoint_batch_evaluation_report_reserve_bytes": 1024**3,
                "new_report_transient_cap_bytes": report_reserve,
                "aggregate_ceiling_bytes": whole_pilot_ceiling,
                "aggregate_cap_bytes": 12 * 1024**3,
                "headroom_bytes": 12 * 1024**3 - whole_pilot_ceiling},
            "hashes": {**actual, "stage1_protocol_sha256": protocol_hash,
                       "protocol_amendment_sha256": digest(amendment_path), **lineage_hashes},
            "reset_plan": reset_plan,
            "planned_frames_if_all_pinned_snapshots_load": sum(x["snapshot_load_native_frames"] + x["fallback_route_replay_frames"] for x in reset_plan) + 21 + ROW_COUNT,
            "native_library_loaded": False, "real_native_frames": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "replay"), default="preflight")
    parser.add_argument("--config", default="training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json")
    parser.add_argument("--source", default="training/level1_full_route_recovery_replay_20261023/training_run/joint_training_data.npz")
    parser.add_argument("--protocol-amendment", default="analysis/level1_dqfd_transition_builder_implementation_20261024/protocol_amendment.json")
    parser.add_argument("--output", required=True, help="fresh output root; preflight only checks that it does not exist")
    parser.add_argument("--frame-cap", type=int, required=True)
    parser.add_argument("--artifact-cap-bytes", type=int, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    parser.add_argument("--adapter", help="replay-only module:function factory; imports lazily")
    parser.add_argument("--resource-receipt", help="fresh CPU replay host/disk launch-gate JSON; replay-only")
    parser.add_argument("--receipt-dir", help="separate parent-side lifecycle/error receipt directory")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    lifecycle = {"phase": "preflight", "native_loading": "not_attempted", "adapter_constructed": False,
                 "trajectory_creation": "not_started", "lineage_collection": "not_started",
                 "native_frames": None, "frame_status": "unknown_until_adapter_ledger"}
    adapter = None
    live_wall_started_at = time.monotonic() if args.mode == "replay" and not args._worker else None
    receipt_dir = Path(args.receipt_dir) if args.receipt_dir else None
    if receipt_dir is not None and not receipt_dir.is_absolute():
        receipt_dir = ROOT / receipt_dir
    try:
        result = preflight(args)
        if output.exists():
            raise TransitionError(f"output root is occupied: {output}")
        if args.mode == "replay":
            if not args._worker:
                if not args.receipt_dir:
                    raise TransitionError("supervised replay requires a fresh --receipt-dir for bounded receipts/logs")
                report_dir = Path(args.receipt_dir)
                if not report_dir.is_absolute():
                    report_dir = ROOT / report_dir
                child_args = []
                skip_value = False
                for value in raw_args:
                    if skip_value:
                        skip_value = False
                        continue
                    if value in ("--resource-receipt", "--_worker"):
                        if value == "--resource-receipt": skip_value = True
                        continue
                    child_args.append(value)
                launch_receipt = report_dir / "reconstruction_resource_launch_receipt.json"
                child_args.extend(["--resource-receipt", str(launch_receipt), "--_worker"])
                child = [sys.executable, str(ROOT / "run.py"), "prepare_level1_dqfd_transitions", *child_args]
                result = supervise_replay(child, cwd=ROOT, report_dir=report_dir, output=output,
                    frame_cap=args.frame_cap, artifact_cap=args.artifact_cap_bytes,
                    wall_cap=args.wall_cap_seconds, started_at=live_wall_started_at)
                print(json.dumps(result, indent=2, sort_keys=True))
                return 0
            if not args.adapter:
                raise TransitionError("replay mode requires an explicit adapter factory")
            if not args.resource_receipt:
                raise TransitionError("replay mode requires a fresh host/disk launch resource receipt")
            receipt_path = ROOT / args.resource_receipt
            resource = json.loads(receipt_path.read_text(encoding="utf-8"))
            lifecycle["attempt_id"] = str(resource.get("attempt_id", ""))
            if resource.get("phase") != "cpu_source_reconstruction" or resource.get("status") != "passed":
                raise TransitionError("resource receipt must be passed for cpu_source_reconstruction")
            measured = datetime.fromisoformat(str(resource.get("measured_at_utc", "")).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - measured.astimezone(timezone.utc)).total_seconds()
            if age < 0 or age > 300:
                raise TransitionError("host/disk launch receipt is future-dated or older than 300 seconds")
            if int(resource.get("available_ram_bytes", 0)) < 8 * 1024**3:
                raise TransitionError("CPU replay launch requires at least 8 GiB available RAM")
            if int(resource.get("commit_headroom_bytes", 0)) < 12 * 1024**3:
                raise TransitionError("CPU replay launch requires at least 12 GiB commit headroom")
            if int(resource.get("disk_free_bytes", 0)) < args.artifact_cap_bytes + 1024**3:
                raise TransitionError("free disk must cover the complete artifact cap plus 1 GiB reserve")
            if not resource.get("disk_target"):
                raise TransitionError("resource receipt must identify the measured destination volume")
            output_drive = os.path.splitdrive(str(output))[0].casefold()
            if str(resource["disk_target"]).casefold() != output_drive:
                raise TransitionError("resource receipt disk target does not match output volume")
            lifecycle["phase"] = "native_construction"
            lifecycle["native_loading"] = "attempted"
            module_name, function_name = args.adapter.split(":", 1)
            factory = getattr(importlib.import_module(module_name), function_name)
            adapter = factory(config=ROOT / args.config, archive=ROOT / args.source,
                              frame_cap=args.frame_cap, artifact_cap=args.artifact_cap_bytes,
                              wall_cap=args.wall_cap_seconds,
                              resource_sample_path=receipt_dir / "runtime_resources.jsonl" if receipt_dir else None,
                              output_root=output)
            lifecycle["native_loading"] = "confirmed"
            lifecycle["adapter_constructed"] = True
            lifecycle["phase"] = "trajectory_creation"
            lifecycle["trajectory_creation"] = "attempted"
            trajectories = adapter.trajectories()
            lifecycle["trajectory_creation"] = "complete"
            lifecycle["phase"] = "lineage_collection"
            lifecycle["lineage_collection"] = "attempted"
            lineage = {**result["hashes"], "reset_plan": result["reset_plan"], **adapter.lineage()}
            lifecycle["lineage_collection"] = "complete"
            code_paths = [ROOT / p for p in (
                "src/prince_valiant_rl/training/dqfd_transitions.py",
                "src/prince_valiant_rl/training/dqfd_level1_adapter.py",
                "src/prince_valiant_rl/training/dqfd_receipts.py",
                "src/prince_valiant_rl/training/dqfd_supervision.py",
                "scripts/prepare_level1_dqfd_transitions.py")]
            identity = {"schema": "level1_dqfd_runtime_identity_v1",
                "attempt_id": str(resource["attempt_id"]), "input_hashes": result["hashes"],
                "tested_code_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in code_paths},
                "runtime_lineage": lineage,
                "verified_before_first_dispatch": True,
                "identity_contract": "observer DLL identity is separate from base codec/ROM/route/config identities"}
            identity_bytes = (json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            if len(identity_bytes) > 64 * 1024:
                raise TransitionError("immutable runtime identity receipt exceeds 64 KiB")
            identity_path = receipt_dir / "identity_receipt.json"
            with identity_path.open("xb") as identity_stream:
                identity_stream.write(identity_bytes)
                identity_stream.flush()
                os.fsync(identity_stream.fileno())
            identity["identity_receipt_sha256"] = hashlib.sha256(identity_bytes).hexdigest()
            lifecycle["identity_receipt"] = {"path": str(identity_path),
                "sha256": identity["identity_receipt_sha256"]}
            lifecycle["phase"] = "replay_and_finalization"
            result = build_cache(trajectories=trajectories, adapter=adapter, output=output,
                                 frame_cap=args.frame_cap, artifact_cap=args.artifact_cap_bytes,
                                 wall_cap=args.wall_cap_seconds, lineage=lineage, identity_receipt=identity,
                                 progress_path=receipt_dir / "progress.jsonl" if receipt_dir else None,
                                 attempt_id=str(resource["attempt_id"]))
            lifecycle["phase"] = "complete"
            lifecycle["native_frames"] = result.get("native_frames")
            lifecycle["frame_status"] = "measured"
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        child_failure = None
        if receipt_dir is not None:
            output_log = receipt_dir / "replay.stdout.log"
            if output_log.is_file():
                try:
                    child_failure = json.loads(output_log.read_text(encoding="utf-8"))
                except Exception:
                    child_failure = None
        if isinstance(child_failure, dict) and isinstance(child_failure.get("lifecycle"), dict):
            lifecycle.update(child_failure["lifecycle"])
            lifecycle["parent_observed_child_failure"] = True
        if lifecycle["phase"] == "native_construction" and lifecycle["native_loading"] == "attempted":
            lifecycle["native_loading"] = "unknown"
        cleanup_error = None
        if adapter is not None and not getattr(adapter, "_closed_by_builder", False):
            try:
                adapter.close()
                adapter._closed_by_builder = True
            except Exception as cleanup_exc:
                cleanup_error = repr(cleanup_exc)[:1000]
        frame_ledger = getattr(adapter, "frame_ledger", None) if adapter is not None else None
        if isinstance(frame_ledger, dict):
            lifecycle["native_frames"] = sum(int(x.get("completed", 0)) for x in frame_ledger.values())
            lifecycle["uncertain_native_frames"] = sum(int(x.get("uncertain", 0)) for x in frame_ledger.values())
            lifecycle["frame_ledger"] = frame_ledger
            lifecycle["frame_status"] = "measured_with_uncertainty" if lifecycle["uncertain_native_frames"] else "measured"
        payload = {"status": "preflight_or_replay_failed", "phase": lifecycle["phase"],
                   "error": str(exc)[:1200], "cleanup_error": cleanup_error, "lifecycle": lifecycle,
                   "data_ready": False, "native_library_loaded": lifecycle["native_loading"] == "confirmed",
                   "native_loading": lifecycle["native_loading"], "native_frames": lifecycle["native_frames"]}
        if isinstance(child_failure, dict):
            payload["child_failure"] = child_failure
        if receipt_dir is not None:
            try:
                _write_parent_failure_receipts(receipt_dir, lifecycle, payload)
            except Exception as receipt_exc:
                payload["parent_receipt_error"] = repr(receipt_exc)[:500]
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
