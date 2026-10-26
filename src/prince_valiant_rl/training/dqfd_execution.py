"""Bounded DQfD learner and fixed-reset collector orchestration primitives.

The module deliberately has no environment/native imports. A live caller must
provide process-owned environment adapters after its own explicit phase gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .dqfd import (OnlineCadence, dqfd_objective, optimizer_step,
                   paired_targets)
from .temporal_models import window_tensors


def state_hash(value: Any) -> str:
    """Stable digest for tensors, numpy values, and nested state dictionaries."""
    digest = hashlib.sha256()
    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode()); digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(array.dtype.str.encode()); digest.update(repr(array.shape).encode()); digest.update(array.tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
        else:
            digest.update(json.dumps(item, sort_keys=True, separators=(",", ":"), default=str).encode())
    visit(value)
    return digest.hexdigest()


def _tensor(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    result = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite learner batch field")
    return result


def _stack_windows(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, np.ndarray]:
    windows = [row[key] for row in rows]
    fields = set(windows[0])
    if any(set(window) != fields for window in windows):
        raise ValueError("mixed replay window schemas differ")
    return {name: np.stack([window[name] for window in windows]) for name in windows[0]}


def _online_row(store: Any, slot: int, generation: int) -> dict[str, Any]:
    row = store.gather_row(int(slot), int(generation))
    return {"current": row["pre"], "following1": row["following1"],
            "following10": row["following10"], "actions": row["action"],
            "action_masks": np.ones(17, np.bool_), "returns1": row["reward1"],
            "durations1": row["duration1"], "discounts1": row["discount1"],
            "bootstrap1": row["bootstrap1"], "returns10": row["reward10"],
            "durations10": row["duration10"], "discounts10": row["discount10"],
            "bootstrap10": row["bootstrap10"], "raw_masks": row["raw_mask"],
            "is_demo": False, "sources": row["source"], "valid": True,
            "generation": row["generation"]}


def gather_mixed_batch(demo_index: Any, online_store: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    """Gather exactly in PER order; never materializes more than one batch."""
    ids = np.asarray(sample["ids"], dtype=np.int64)
    demo_flags = np.asarray(sample["is_demo"], dtype=np.bool_)
    generations = np.asarray(sample["generation"], dtype=np.int64)
    if ids.ndim != 1 or not 1 <= len(ids) <= 128 or demo_flags.shape != ids.shape or generations.shape != ids.shape:
        raise ValueError("invalid mixed replay sample")
    positions = np.flatnonzero(demo_flags)
    demo = demo_index.gather(ids[positions], probabilities=np.asarray(sample["probability"])[positions],
                             weights=np.asarray(sample["weights"])[positions]) if len(positions) else None
    demo_at = {int(position): index for index, position in enumerate(positions)}
    online_at = {int(position): index for index, position in enumerate(np.flatnonzero(~demo_flags))}
    online_positions = np.flatnonzero(~demo_flags)
    if len(online_positions) and online_store is None:
        raise ValueError("online rows are unavailable in demo-only mode")
    online_rows = [_online_row(online_store, ids[position], generations[position])
                   for position in online_positions]
    rows: list[dict[str, Any]] = []
    def row_value(value: Any, index: int) -> Any:
        if isinstance(value, Mapping):
            return {key: row_value(child, index) for key, child in value.items()}
        if isinstance(value, np.ndarray) and value.ndim:
            return value[index]
        return value
    for position in range(len(ids)):
        if demo_flags[position]:
            j = demo_at[position]
            rows.append({key: row_value(value, j)
                         for key, value in demo.items()
                         if key not in ("measurement", "sources")})
            rows[-1]["sources"] = demo["sources"][j]
            rows[-1]["valid"] = bool(demo["valid"][j])
        else:
            rows.append(online_rows[online_at[position]])
    if not rows or not all(bool(row["valid"]) for row in rows):
        raise ValueError("invalid replay row in learner batch")
    names = ("current", "following1", "following10")
    batch: dict[str, Any] = {name: _stack_windows(rows, name) for name in names}
    scalar_keys = ("actions", "action_masks", "returns1", "durations1", "discounts1", "bootstrap1",
                   "returns10", "durations10", "discounts10", "bootstrap10", "raw_masks")
    for key in scalar_keys:
        batch[key] = np.stack([np.asarray(row[key]) for row in rows])
    batch.update(is_demo=demo_flags.copy(), weights=np.asarray(sample["weights"], np.float32).copy(),
                 probability=np.asarray(sample["probability"], np.float64).copy(),
                 sources=[row["sources"] for row in rows], ids=ids.copy(),
                 generation=generations.copy(), valid=np.ones(len(rows), np.bool_))
    return batch


@dataclass
class LearnerCounters:
    successful_steps: int = 0
    attempted_steps: int = 0
    applied_steps: int = 0
    target_copies: int = 0
    failed_after_step: int = 0
    unknown_step_outcomes: int = 0
    phase: str = ""
    phase_identity: str = ""
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


class DQfDBatchLearner:
    """Production single-batch DQfD update over the existing replay APIs."""
    def __init__(self, online: torch.nn.Module, target: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, demo_index: Any, online_store: Any,
                 rng: np.random.Generator, *, successful_steps: int = 0,
                 phase: str = "", phase_identity: str = "", diagnostics_limit: int = 256,
                 demo_only: bool = False):
        self.online, self.target, self.optimizer = online, target, optimizer
        self.demo_index, self.store, self.rng = demo_index, online_store, rng
        self.demo_only = bool(demo_only)
        if self.demo_only and self.store is not None:
            raise ValueError("demo-only learner must not receive an online frame store")
        if self.demo_only and int(successful_steps) != 0:
            raise ValueError("demo-only pretraining requires a fresh global update counter")
        if not self.demo_only and self.store is None:
            raise ValueError("mixed replay learner requires an online store")
        self.device = next(online.parameters()).device
        self.counters = LearnerCounters(successful_steps=int(successful_steps),
            phase=phase, phase_identity=phase_identity)
        self.diagnostics_limit = int(diagnostics_limit)
        self._optimizer_step = optimizer_step
        self._optimizer_mutated = False
        self._optimizer_called = False
        self._optimizer_pre_hook = optimizer.register_step_pre_hook(
            lambda _optimizer, _args, _kwargs: setattr(self, "_optimizer_called", True))
        self._optimizer_hook = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: setattr(self, "_optimizer_mutated", True))

    def update(self, beta: float, *, receipt_fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        if not callable(receipt_fn):
            raise ValueError("a durable update receipt callback is required")
        token = None
        if self.demo_only:
            priorities = np.asarray(self.demo_index._priority, dtype=np.float64)
            if priorities.ndim != 1 or len(priorities) < 1 or not np.isfinite(priorities).all() or np.any(priorities < 0):
                raise ValueError("protected demo PER priorities are invalid")
            mass = (priorities + 1.0) ** .4
            probabilities = mass / mass.sum()
            ids = self.rng.choice(len(probabilities), size=128, replace=True, p=probabilities)
            sampled_probability = probabilities[ids]
            weights = (len(probabilities) * sampled_probability) ** (-float(beta))
            weights = (weights / weights.max()).astype(np.float32)
            sample = {"ids": ids.astype(np.int64), "generation": np.full(128, -1, np.int64),
                "is_demo": np.ones(128, np.bool_), "probability": sampled_probability,
                "weights": weights}
        else:
            sample, token = self.store.sample(128, self.rng, float(beta))
        stepped = False
        try:
            batch = gather_mixed_batch(self.demo_index, self.store, sample)
            # Validate all scalar and full active-window tensors before the optimizer is touched.
            fields = ("actions", "action_masks", "returns1", "discounts1", "bootstrap1",
                      "returns10", "discounts10", "bootstrap10", "weights")
            tensors = {key: _tensor(batch[key], self.device,
                torch.bool if key in ("action_masks", "bootstrap1", "bootstrap10") else
                torch.long if key == "actions" else torch.float32) for key in fields}
            current = window_tensors(batch["current"], self.device, batch=True)
            following1 = window_tensors(batch["following1"], self.device, batch=True)
            following10 = window_tensors(batch["following10"], self.device, batch=True)
            if tensors["actions"].shape != (128,) or tensors["action_masks"].shape != (128, 17):
                raise ValueError("learner batch must have 128 rows and 17 action masks")
            for window in (current, following1, following10):
                if not window or any(value.shape[0] != 128 for value in window.values()):
                    raise ValueError("active field transfer is incomplete or has a wrong batch size")
            q = self.online(current)
            y1, y10 = paired_targets(self.online, self.target, following1, following10,
                tensors["action_masks"], tensors["action_masks"], tensors["returns1"],
                tensors["discounts1"], tensors["bootstrap1"], tensors["returns10"],
                tensors["discounts10"], tensors["bootstrap10"])
            loss, parts = dqfd_objective(q, tensors["actions"], y1, y10, tensors["weights"],
                _tensor(batch["is_demo"], self.device, torch.bool),
                (parameter for parameter in self.online.parameters() if parameter.requires_grad),
                action_mask=tensors["action_masks"])
            # Diagnostics and PER values are finite before optimizer mutation.
            td1 = parts["td1"].detach().cpu().numpy().astype(np.float64)
            td10 = parts["td10"].detach().cpu().numpy().astype(np.float64)
            scalar_loss = float(loss.detach().cpu())
            if not math.isfinite(scalar_loss) or not np.isfinite(td1).all() or not np.isfinite(td10).all():
                raise FloatingPointError("nonfinite learner diagnostics")
            self.counters.attempted_steps += 1
            self._optimizer_mutated = False
            self._optimizer_called = False
            next_step, grad_norm, synced = self._optimizer_step(self.online, self.target,
                self.optimizer, loss, self.counters.successful_steps,
                applied_callback=lambda _step: setattr(self.counters, "applied_steps", self.counters.applied_steps + 1),
                target_copy_callback=lambda _step: setattr(self.counters, "target_copies", self.counters.target_copies + 1))
            stepped = True
            self.counters.successful_steps = next_step
            diagnostic = {"phase": self.counters.phase, "phase_identity": self.counters.phase_identity,
                "successful_step": next_step, "loss": scalar_loss, "gradient_norm": grad_norm,
                "target_copy": bool(synced), "demo_fraction": float(np.mean(batch["is_demo"])),
                "td1_abs_mean": float(np.mean(np.abs(td1))), "td10_abs_mean": float(np.mean(np.abs(td10))),
                "margin_mean": float(parts["margin"].mean().detach().cpu()),
                "l2": float(parts["regularizer"].detach().cpu()),
                "q_abs_mean": float(q.detach().abs().mean().cpu())}
            if len(self.counters.diagnostics) < self.diagnostics_limit:
                self.counters.diagnostics.append(diagnostic)
            if self.demo_only:
                self.demo_index.update_priority(sample["ids"], np.maximum(np.abs(td1), np.abs(td10)) + .001)
            else:
                self.store.update_priorities(sample, td1, td10)
            diagnostic["parameter_sha256"] = state_hash(self.online.state_dict())
            diagnostic["target_sha256"] = state_hash(self.target.state_dict())
            diagnostic["optimizer_sha256"] = state_hash(self.optimizer.state_dict())
            diagnostic["rng_sha256"] = state_hash(self.rng.bit_generator.state)
            receipt_fn(diagnostic)
            return diagnostic
        except BaseException:
            if stepped or self._optimizer_mutated:
                self.counters.failed_after_step += 1
            elif self._optimizer_called:
                self.counters.unknown_step_outcomes += 1
            raise
        finally:
            if token is not None:
                self.store.release_lease(token)

    def close(self) -> None:
        """Release optimizer hooks at a phase boundary."""
        for name in ("_optimizer_pre_hook", "_optimizer_hook"):
            handle = getattr(self, name, None)
            if handle is not None:
                handle.remove()
                setattr(self, name, None)


class FixedBCollector:
    """Six-owner fixed-B round driver over injected process-env adapters.

    Each adapter must expose reset_trusted_b() -> (window, identities),
    action(window, rng, epsilon) -> (catalog_action, raw_mask), step(raw_mask),
    and close(). It is intentionally not a CLI fake backend.
    """
    def __init__(self, envs: Sequence[Any], store: Any, cadence: OnlineCadence,
                 canonical_window: Mapping[str, Any], canonical_ids: Sequence[Sequence[Any]],
                 rng: np.random.Generator, *, seed: int = 20261024,
                 episode_sink: Callable[[dict[str, Any]], Any] | None = None):
        if len(envs) != 6 or cadence.workers != 6:
            raise ValueError("pilot collector contract is fixed at six owners")
        if not callable(episode_sink):
            raise ValueError("an append-only episode outcome sink is required")
        self.envs, self.store, self.cadence = list(envs), store, cadence
        self.canonical_window, self.canonical_ids = canonical_window, canonical_ids
        self.rng, self.seed = rng, int(seed)
        self.episodes = [0] * 6
        self.episode_sink = episode_sink
        self.finished_count = 0
        self.finished_by_owner = [0] * 6
        self.episode_length_by_owner = [{"min": None, "max": 0, "sum": 0} for _ in range(6)]
        self.terminal_reasons: dict[str, int] = {}
        self.first_success_frame: int | None = None
        self.dispatch_count = 0
        self.dispatch_digest = hashlib.sha256()

    @staticmethod
    def epsilon(frames: int) -> float:
        return .20 + (.02 - .20) * min(1., max(0., frames / 184320.) )

    def _reset(self, owner: int) -> dict[str, Any]:
        window, identities = self.envs[owner].reset_trusted_b()
        # Store performs byte/dtype/validity/controller-history comparison before mutation.
        self.store.reset(owner, f"owner{owner}:episode{self.episodes[owner]}", window,
                         identities, source="trusted_B")
        return {"window": window, "identities": identities}

    def run(self, *, on_update: Callable[[float], Any] | None = None) -> dict[str, Any]:
        states = []
        try:
            self.store.set_trusted_reset_history(self.canonical_window, self.canonical_ids)
            states = [self._reset(owner) for owner in range(6)]
            while self.cadence.frames < self.cadence.cap:
                width = self.cadence.dispatch_width()
                if width != 6: raise RuntimeError("final aggregate frame remainder cannot be dispatched as six whole actions")
                start = self.cadence.frames
                actions = []
                for owner in range(6):
                    frame_offset = start + owner
                    epsilon = self.epsilon(frame_offset)
                    action, raw_mask = self.envs[owner].action(states[owner]["window"], self.rng, epsilon)
                    if not 0 <= int(action) < 17: raise ValueError("action outside 17-action catalog")
                    actions.append((int(action), int(raw_mask), epsilon, frame_offset))
                ended: list[int] = []
                for owner, (action, raw_mask, epsilon, frame_offset) in enumerate(actions):
                    transition = self.envs[owner].step(raw_mask)
                    if transition.get("observation") is None: raise ValueError("missing actual final/following observation")
                    info = transition["info"]
                    duration = int(transition["duration"])
                    if duration != 1: raise RuntimeError("vector credit schedule requires one native frame per action")
                    boundary = bool(info["natural_terminal"] or info["time_limit"])
                    bootstrap = bool(info["bootstrap_valid"])
                    self.store.add(owner, action=action, raw_mask=raw_mask,
                        reward=float(transition["reward"]), duration=duration,
                        post_window=transition["observation"], post_identities=transition["identities"],
                        boundary=boundary, natural_terminal=bool(info["natural_terminal"]), bootstrap=bootstrap)
                    event = {"owner": owner, "episode": self.episodes[owner], "frame_offset": frame_offset,
                        "action": action, "raw_mask": raw_mask, "epsilon": epsilon, "duration": duration}
                    self.dispatch_digest.update(json.dumps(event, sort_keys=True, separators=(",", ":")).encode())
                    self.dispatch_count += 1
                    states[owner] = {"window": transition["observation"], "identities": transition["identities"]}
                    if boundary:
                        outcome = dict(owner=owner, episode=self.episodes[owner],
                            frames=int(self.envs[owner].episode_frames), reason=str(info.get("end_reason") or "unknown"),
                            terminal=bool(info["natural_terminal"]), bootstrap=bootstrap,
                            aggregate_frame=frame_offset + duration)
                        self.episode_sink(outcome)
                        self.finished_count += 1
                        self.finished_by_owner[owner] += 1
                        aggregate = self.episode_length_by_owner[owner]
                        aggregate["min"] = outcome["frames"] if aggregate["min"] is None else min(aggregate["min"], outcome["frames"])
                        aggregate["max"] = max(aggregate["max"], outcome["frames"])
                        aggregate["sum"] += outcome["frames"]
                        self.terminal_reasons[outcome["reason"]] = self.terminal_reasons.get(outcome["reason"], 0) + 1
                        if outcome["reason"] == "level_completed" and self.first_success_frame is None:
                            self.first_success_frame = outcome["aggregate_frame"]
                        ended.append(owner)
                events = self.cadence.complete_round(6)
                if on_update:
                    first_event_frame = self.cadence.next_event - events * self.cadence.cadence
                    for event_index in range(events):
                        event_frame = first_event_frame + event_index * self.cadence.cadence
                        on_update(.6 + .4 * min(1., event_frame / self.cadence.cap))
                if self.cadence.frames == self.cadence.cap:
                    continue
                for owner in ended:
                    self.episodes[owner] += 1
                    states[owner] = self._reset(owner)
            dropped = self.store.finish_budget()
            return {"frames": self.cadence.frames, "update_events": self.cadence.events,
                "finished_episodes": self.finished_count, "finished_by_owner": self.finished_by_owner,
                "episode_length_by_owner": self.episode_length_by_owner,
                "terminal_reasons": self.terminal_reasons,
                "censored_owners": sum(1 for count in self.finished_by_owner if count == 0),
                "pending_rows_dropped": dropped, "first_success_frame": self.first_success_frame,
                "dispatch_count": self.dispatch_count,
                "dispatch_order_sha256": self.dispatch_digest.hexdigest()}
        finally:
            active_error = sys.exc_info()[1]
            cleanup_errors = []
            for env in self.envs:
                try: env.close()
                except Exception as exc: cleanup_errors.append(exc)
            if getattr(self.store, "leases", {}):
                for token in list(self.store.leases):
                    try: self.store.release_lease(token)
                    except Exception as exc: cleanup_errors.append(exc)
            if hasattr(self.store, "close"):
                try: self.store.close()
                except Exception as exc: cleanup_errors.append(exc)
            if cleanup_errors:
                if active_error is not None:
                    active_error.add_note("collector cleanup failures: " + "; ".join(repr(e)[:200] for e in cleanup_errors))
                else:
                    raise RuntimeError("collector cleanup failed: " + "; ".join(repr(e)[:200] for e in cleanup_errors))
