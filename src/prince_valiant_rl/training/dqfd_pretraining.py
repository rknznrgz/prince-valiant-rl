"""Demo-only pretraining loop and predeclared periodic demo/retention protocol."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import torch

from .dqfd_execution import state_hash


def learner_state_hashes(learner: Any) -> dict[str, str]:
    return {"q": state_hash(learner.online.state_dict()), "target": state_hash(learner.target.state_dict()),
        "optimizer": state_hash(learner.optimizer.state_dict()), "rng": state_hash({
            "numpy": learner.rng.bit_generator.state, "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}),
        "per": state_hash(learner.demo_index._priority if learner.demo_only else
            {"demo": learner.demo_index._priority, "online_base": learner.store.per.online_base,
             "online_generation": learner.store.per.online_generation})}


def _save_milestone(learner: Any, path: Path, update: int) -> dict[str, str]:
    if int(learner.counters.successful_steps) != int(update):
        raise ValueError("milestone checkpoint counter must equal the exact successful update")
    if path.exists(): raise FileExistsError(f"milestone checkpoint is occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(getattr(learner, "model_metadata", {}))
    required = ("architecture", "shapes", "actions", "model_config", "actor_sha256",
                "config_sha256", "manifest_sha256", "READY_sha256", "identity_sha256",
                "optimizer_contract")
    if any(key not in metadata for key in required):
        raise ValueError("milestone learner is missing pinned Q architecture/config metadata")
    payload = {"format": "dqfd_q_v1", "successful_update": int(update), "metadata": metadata,
        "online": learner.online.state_dict(), "target": learner.target.state_dict(),
        "optimizer": learner.optimizer.state_dict(), "rng": learner.rng.bit_generator.state,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "demo_priority": learner.demo_index._priority.copy(),
        "counters": {name: int(getattr(learner.counters, name)) for name in
            ("successful_steps", "attempted_steps", "applied_steps", "target_copies",
             "failed_after_step", "unknown_step_outcomes")},
        "state_hashes": learner_state_hashes(learner)}
    with path.open("xb") as stream:
        torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = {"status": "committed", "format": "dqfd_q_v1", "successful_update": int(update),
        "checkpoint_sha256": digest, "state_hashes": payload["state_hashes"],
        "metadata": metadata}
    receipt_bytes = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    pending_receipt = receipt_path.with_name(receipt_path.name + ".pending")
    with pending_receipt.open("xb") as stream:
        stream.write(receipt_bytes); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending_receipt, receipt_path)
    return {"checkpoint_path": str(path), "checkpoint_sha256": digest,
            "checkpoint_receipt_path": str(receipt_path),
            **payload["state_hashes"]}


def load_committed_milestone(path: Path, *, expected_update: int, expected_pins: Mapping[str, str],
                             expected_config_hash: str, model_factory: Callable[..., Any] | None = None,
                             map_location: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Reject partial, foreign, or modified checkpoints before model creation."""
    path = Path(path)
    checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "committed" or receipt.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("milestone committed receipt is missing or mismatched")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "dqfd_q_v1" or int(payload.get("successful_update", -1)) != int(expected_update):
        raise ValueError("milestone format/update identity mismatch")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("config_sha256") != expected_config_hash:
        raise ValueError("milestone config identity missing or foreign")
    for key, value in expected_pins.items():
        if metadata.get(key) != value:
            raise ValueError(f"milestone pin mismatch: {key}")
    if not all(key in payload for key in ("online", "target", "optimizer", "rng", "torch_cpu_rng",
                                           "torch_cuda_rng", "demo_priority", "counters", "state_hashes")):
        raise ValueError("milestone state is partial")
    online_state, target_state = payload["online"], payload["target"]
    if (not isinstance(online_state, dict) or not isinstance(target_state, dict)
            or set(online_state) != set(target_state)
            or any(tuple(online_state[key].shape) != tuple(target_state[key].shape) for key in online_state)):
        raise ValueError("milestone online/target architecture state differs")
    priorities = payload["demo_priority"]
    if (not hasattr(priorities, "shape") or tuple(priorities.shape) != (21599,)
            or not torch.isfinite(torch.as_tensor(priorities)).all() or (priorities < 0).any()):
        raise ValueError("milestone protected demo PER state is missing or invalid")
    if not isinstance(payload["optimizer"], dict) or not isinstance(payload["optimizer"].get("param_groups"), list):
        raise ValueError("milestone Adam state is missing or malformed")
    if metadata.get("optimizer_contract") != {"name": "Adam", "lr": 1e-4, "eps": 1e-8,
            "betas": [.9, .999], "weight_decay": 0.0}:
        raise ValueError("milestone optimizer configuration is foreign")
    from .dqfd_execution import state_hash
    exact = {"q": state_hash(payload["online"]), "target": state_hash(payload["target"]),
        "optimizer": state_hash(payload["optimizer"]), "rng": state_hash({"numpy": payload["rng"],
            "torch_cpu": payload["torch_cpu_rng"], "torch_cuda": payload["torch_cuda_rng"]}),
        "per": state_hash(payload["demo_priority"])}
    if payload["state_hashes"] != exact:
        raise ValueError("milestone exact state hash mismatch")
    if receipt.get("state_hashes") != exact or receipt.get("metadata") != metadata:
        raise ValueError("milestone receipt state/metadata identity differs from payload")
    counters = payload["counters"]
    if (int(counters.get("successful_steps", -1)) != int(expected_update)
            or int(counters.get("attempted_steps", -1)) != int(expected_update)
            or int(counters.get("applied_steps", -1)) != int(expected_update)
            or int(counters.get("target_copies", -1)) != expected_update // 1000
            or int(counters.get("failed_after_step", -1)) != 0
            or int(counters.get("unknown_step_outcomes", -1)) != 0):
        raise ValueError("milestone counters/target cadence mismatch")
    if model_factory is None:
        from .dqfd import DuelingQ
        model_factory = DuelingQ
    # All identity and completeness checks precede construction.
    model = model_factory(metadata["shapes"], int(metadata["actions"]), metadata["model_config"])
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    if not payload["state_hashes"].get("q"):
        raise ValueError("milestone exact state hashes missing")
    return model, {"checkpoint_sha256": checkpoint_hash, "metadata": metadata,
                   "state_hashes": payload["state_hashes"], "successful_update": int(expected_update)}


def _rank(candidate: Mapping[str, Any]) -> tuple[Any, ...] | None:
    if (candidate.get("status") != "complete" or candidate.get("runtime_identity_matched") is not True
            or candidate.get("reset_provenance_verified") is not True
            or candidate.get("terminal_valid") is not True):
        return None
    reward = candidate.get("cumulative_applied_reward")
    video_bytes = candidate.get("video_bytes")
    update = candidate.get("update")
    clears = candidate.get("verified_objective_clear_identities")
    end_reason = candidate.get("end_reason")
    if (isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward))
            or isinstance(video_bytes, bool) or not isinstance(video_bytes, int)
            or not 0 < video_bytes <= 100 * 1024 * 1024
            or isinstance(update, bool) or not isinstance(update, int) or update not in (1000, 2000)
            or not isinstance(end_reason, str) or not end_reason
            or not isinstance(clears, (list, tuple)) or any(not isinstance(item, str) or not item for item in clears)
            or len(clears) > 256):
        return None
    # Natural completion, distinct verified clears, measured configured reward,
    # and earlier update on exact tie. Score availability is retained as a
    # qualification field but does not silently alter the precommitted order.
    return (int(candidate.get("end_reason") == "level_completed"),
        len(set(candidate.get("verified_objective_clear_identities", []))),
        float(reward), -update)


def _retention_passed(gate: Mapping[str, Any]) -> bool:
    return _rank(gate) is not None and gate.get("end_reason") == "level_completed"


def run_demo_pretraining(learner: Any, *, output_dir: Path,
                         episode_fn: Callable[[int, Path, str], Mapping[str, Any]],
                         update_receipt_fn: Callable[[dict[str, Any]], Any],
                         checkpoint_fn: Callable[[Any, Path, int], Mapping[str, Any]] = _save_milestone,
                         checkpoint_receipt_fn: Callable[[dict[str, Any]], Any] | None = None,
                         on_update: Callable[[int], Any] | None = None,
                         boundary_fn: Callable[[str], Any] | None = None,
                         clock: Callable[[], float] = time.monotonic,
                         started_at: float | None = None,
                         execution_budget: Mapping[str, Any]) -> dict[str, Any]:
    """Run exactly updates 1..2000, pausing at 1000/2000 for isolated demos.

    ``episode_fn`` must use the real reset-389 evaluator. Its failure stops the
    phase. Evaluation callbacks cannot mutate learner state: all five handoff
    hashes are checked before and after each pause.
    """
    if not learner.demo_only or learner.store is not None:
        raise ValueError("pretraining requires the actual demo-only learner without an online store")
    if learner.counters.successful_steps != 0 or learner.counters.attempted_steps != 0:
        raise ValueError("pretraining must start with fresh counters; resume/retry is prohibited")
    from .dqfd_study_budget import validate_loop_limits
    validate_loop_limits(execution_budget)
    whole_started = clock() if started_at is None else float(started_at)
    def check_budget(stage: str) -> None:
        elapsed = max(0.0, float(clock()) - whole_started)
        elapsed_ticks = math.ceil(elapsed * 10_000_000)
        if (elapsed_ticks > int(execution_budget["remaining_active_ticks"])
                or elapsed_ticks > int(execution_budget["slice_active_ticks"])):
            raise TimeoutError(f"cumulative/slice execution budget exceeded during {stage}")
        if boundary_fn is not None:
            boundary_fn(stage)
    root = Path(output_dir)
    if root.exists(): raise FileExistsError(f"pretraining phase destination is occupied: {root}")
    root.mkdir(parents=True)
    milestones = []; model_io_seconds = max(0., clock() - whole_started); demo_seconds = 0.; charged_demo_frames = 0
    model_cap = float(execution_budget["model_io_active_seconds"])
    demo_cap = float(execution_budget["demo_encoding_active_seconds"])
    if model_io_seconds > model_cap: raise RuntimeError(f"pretraining/model-I/O active-time proposal exceeded {model_cap:g} seconds during supervised setup")
    check_budget("initialization_complete")
    for update in range(1, 2001):
        check_budget("optimizer_update_enter")
        io_started = clock()
        # Pretraining uses the frozen constant beta=.6.  The .6->1.0 schedule
        # belongs only to the later online phase.
        row = learner.update(.6, receipt_fn=update_receipt_fn)
        if int(row["successful_step"]) != update:
            raise RuntimeError("learner global successful counter skipped or repeated")
        if on_update is not None: on_update(update)
        check_budget("optimizer_update_leave")
        if update not in (1000, 2000):
            model_io_seconds += max(0., clock() - io_started)
            if model_io_seconds > model_cap: raise RuntimeError(f"pretraining/model-I/O active-time proposal exceeded {model_cap:g} seconds")
            continue
        checkpoint_path = root / f"checkpoint_{update:04d}.pt"
        check_budget("checkpoint_write_enter")
        checkpoint = dict(checkpoint_fn(learner, checkpoint_path, update))
        check_budget("checkpoint_readback")
        if checkpoint_receipt_fn is not None:
            checkpoint_receipt_fn({"update": update, "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
                "checkpoint_receipt_path": checkpoint.get("checkpoint_receipt_path"),
                "state_hashes": checkpoint.get("state_hashes", {})})
        model_io_seconds += max(0., clock() - io_started)
        if model_io_seconds > model_cap: raise RuntimeError(f"pretraining/model-I/O active-time proposal exceeded {model_cap:g} seconds")
        before = learner_state_hashes(learner)
        if checkpoint.get("state_hashes", before) != before:
            raise RuntimeError("milestone checkpoint handoff hashes differ from live learner state")
        label = "candidate" if update == 1000 else "fixed2000_gate_candidate"
        demo_started = clock()
        check_budget("demo_encoding_enter")
        evidence = dict(episode_fn(update, checkpoint_path, label))
        check_budget("demo_encoding_leave")
        demo_seconds += max(0., clock() - demo_started)
        if demo_seconds > demo_cap: raise RuntimeError(f"pretraining demo/gate/encoding proposal exceeded {demo_cap:g} seconds")
        after = learner_state_hashes(learner)
        if after != before:
            raise RuntimeError("evaluation mutated training Q/target/Adam/RNG/PER handoff")
        evidence.update(update=update, label=label, checkpoint_sha256=checkpoint["checkpoint_sha256"],
            state_hashes=before, score_availability=evidence.get("score_availability"))
        provenance = evidence.get("reset_provenance", {})
        charged = (int(evidence.get("native_frames", 0))
            + int(evidence.get("initialization_native_frames", 0))
            + int(provenance.get("restoration_native_frames", 0)))
        if charged > 36000: raise RuntimeError("demo episode including initialization/reset advances exceeded 36000 frames")
        evidence["charged_native_frames"] = charged
        charged_demo_frames += charged
        if charged_demo_frames > 108000: raise RuntimeError("pretraining demo/gate aggregate frame proposal exceeded 108000")
        if sum(int(item.get("video_bytes", 0)) for item in [*milestones, evidence]) > 300 * 1024 * 1024:
            raise RuntimeError("pretraining demo/gate aggregate video bundle cap exceeded")
        milestones.append(evidence)
        if update == 2000 and evidence.get("runtime_error"):
            raise RuntimeError("fixed2000 gate stopped on runtime/resource/provenance failure")
    candidates = [(rank, item) for item in milestones if (rank := _rank(item)) is not None]
    best = max(candidates, key=lambda pair: pair[0])[1] if candidates else None
    extra = None
    if best is not None and int(best["update"]) == 1000:
        before = learner_state_hashes(learner)
        demo_started = clock()
        check_budget("demo_encoding_enter")
        extra = dict(episode_fn(1000, root / "checkpoint_1000.pt", "selected_best_demo_after_gate"))
        check_budget("demo_encoding_leave")
        demo_seconds += max(0., clock() - demo_started)
        if demo_seconds > demo_cap: raise RuntimeError(f"pretraining demo/gate/encoding proposal exceeded {demo_cap:g} seconds")
        if learner_state_hashes(learner) != before:
            raise RuntimeError("isolated update-1000 best demo mutated training handoff state")
        provenance = extra.get("reset_provenance", {})
        charged = (int(extra.get("native_frames", 0))
            + int(extra.get("initialization_native_frames", 0))
            + int(provenance.get("restoration_native_frames", 0)))
        if charged > 36000 or charged_demo_frames + charged > 108000:
            raise RuntimeError("extra best demo exceeded its bounded frame allowance")
        extra.update(update=1000, label="selected_best_demo_after_gate", checkpoint_sha256=
            next(item["checkpoint_sha256"] for item in milestones if item["update"] == 1000))
        extra["charged_native_frames"] = charged
        if sum(int(item.get("video_bytes", 0)) for item in [*milestones, extra]) > 300 * 1024 * 1024:
            raise RuntimeError("pretraining demo/gate aggregate video bundle cap exceeded")
    # The slice caps are charged against the original pilot ceiling.  Include
    # loop setup, state hashing, receipt work, and finalization in the model-I/O
    # segment; the enclosing parent watchdog independently covers child startup
    # through process teardown and must not reset between phases.
    check_budget("finalization_enter")
    accounted = max(0., clock() - whole_started)
    unclassified = accounted - model_io_seconds - demo_seconds
    if unclassified > 0:
        model_io_seconds += unclassified
    if model_io_seconds > model_cap: raise RuntimeError(f"pretraining/model-I/O active-time proposal exceeded {model_cap:g} seconds")
    if demo_seconds > demo_cap: raise RuntimeError(f"pretraining demo/gate/encoding proposal exceeded {demo_cap:g} seconds")
    cumulative = (int(execution_budget["prior_charged_ticks"]) + math.ceil(accounted * 10_000_000)) / 10_000_000
    remaining = float(execution_budget["remaining_original_active_seconds"])
    if math.ceil(accounted * 10_000_000) > int(execution_budget["remaining_active_ticks"]):
        raise RuntimeError(f"cumulative active-time ceiling of {execution_budget['original_total_active_seconds']} seconds exceeded")
    gate = milestones[1]
    return {"successful_updates": learner.counters.successful_steps,
        "attempted_updates": learner.counters.attempted_steps, "target_copies": learner.counters.target_copies,
        "milestones": milestones, "extra_best_demo": extra, "best_evaluated_candidate_update":
            int(best["update"]) if best is not None else None,
        "model_io_active_seconds": model_io_seconds, "demo_gate_encoding_active_seconds": demo_seconds,
        "whole_supervised_active_seconds": accounted,
        "remaining_original_active_seconds_before_slice": remaining,
        "cumulative_pilot_active_seconds": cumulative, "charged_demo_native_frames": charged_demo_frames
            + int(extra.get("charged_native_frames", 0)) if extra else charged_demo_frames,
        "fixed2000_retention_passed": _retention_passed(gate),
        "episode_count": len(milestones) + int(extra is not None),
        "state_hashes": learner_state_hashes(learner)}
