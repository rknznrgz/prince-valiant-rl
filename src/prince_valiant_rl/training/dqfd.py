"""Frozen duration-aware DQfD objective and bounded global prioritized replay."""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import math
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .dqn import DuelingQ, masked_argmax
from .temporal_models import window_tensors

N_STEP = 10
ALPHA = .4
PRIORITY_EPSILON = .001
DEMO_BONUS = 1.0
MARGIN = .8


def paired_return(rows: Iterable[dict[str, Any]], gamma: float, n: int = N_STEP) -> dict[str, Any]:
    """Aggregate one owner-local return; boundaries shorten and flush the queue."""
    seq = list(rows)
    if not seq or n < 1 or not 0 < gamma <= 1:
        raise ValueError("invalid duration return input")
    first = seq[0]
    reward, discount, duration, last = 0.0, 1.0, 0, None
    for row in seq[:n]:
        if row.get("owner") != first.get("owner") or row.get("episode") != first.get("episode"):
            raise ValueError("n-step return crossed owner/episode boundary")
        if int(row["duration"]) < 1:
            raise ValueError("native duration must be positive")
        reward += discount * float(row["reward"])
        duration += int(row["duration"])
        discount *= gamma ** int(row["duration"])
        last = row
        if row.get("boundary", False):
            break
    assert last is not None
    return {"reward": reward, "discount": discount,
            "bootstrap": bool(last["bootstrap"]), "duration": duration,
            "following": last["following"], "boundary": bool(last.get("boundary", False))}


def paired_targets(online: nn.Module, target: nn.Module, following1: dict[str, torch.Tensor],
                   following10: dict[str, torch.Tensor], masks1: torch.Tensor,
                   masks10: torch.Tensor, rewards1: torch.Tensor, discounts1: torch.Tensor,
                   bootstrap1: torch.Tensor, rewards10: torch.Tensor,
                   discounts10: torch.Tensor, bootstrap10: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Double-Q selection/evaluation with detached, per-step duration targets."""
    with torch.no_grad():
        for value in (rewards1, discounts1, bootstrap1, rewards10, discounts10, bootstrap10):
            if not torch.isfinite(value).all(): raise FloatingPointError("nonfinite reward/discount/bootstrap target input")
        if (discounts1 < 0).any() or (discounts10 < 0).any(): raise ValueError("negative return discount")
        def calc(window: dict[str, torch.Tensor], mask: torch.Tensor, reward: torch.Tensor,
                 discount: torch.Tensor, bootstrap: torch.Tensor) -> torch.Tensor:
            q_online = online(window)
            action = masked_argmax(q_online, mask)
            q_target = target(window).gather(1, action[:, None]).squeeze(1)
            return reward + discount * bootstrap.to(reward.dtype) * q_target
        return calc(following1, masks1, rewards1, discounts1, bootstrap1), \
            calc(following10, masks10, rewards10, discounts10, bootstrap10)


def dqfd_objective(q: torch.Tensor, actions: torch.Tensor, y1: torch.Tensor,
                   y10: torch.Tensor, weights: torch.Tensor, demo: torch.Tensor,
                   trainable_parameters: Iterable[torch.Tensor], action_mask: torch.Tensor | None = None,
                   margin: float = MARGIN, l2: float = 1e-5) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Batch mean of IS-weighted paired Huber, demo margin, and one L2 term."""
    if q.ndim != 2 or not (q.shape[0] == len(actions) == len(y1) == len(y10) == len(weights) == len(demo)):
        raise ValueError("DQfD objective batch shape mismatch")
    if not torch.isfinite(q).all() or not torch.isfinite(y1).all() or not torch.isfinite(y10).all():
        raise FloatingPointError("nonfinite Q values or targets")
    action_mask = torch.ones_like(q, dtype=torch.bool) if action_mask is None else torch.as_tensor(action_mask, device=q.device, dtype=torch.bool)
    if action_mask.ndim == 1:
        action_mask = action_mask.expand_as(q)
    if action_mask.shape != q.shape or not action_mask.any(-1).all():
        raise ValueError("invalid action mask")
    actions = actions.long()
    if not action_mask.gather(1, actions[:, None]).all():
        raise ValueError("demonstrated action is masked")
    selected = q.gather(1, actions[:, None]).squeeze(1)
    d1, d10 = y1.detach() - selected, y10.detach() - selected
    row1 = F.smooth_l1_loss(selected, y1.detach(), reduction="none")
    row10 = F.smooth_l1_loss(selected, y10.detach(), reduction="none")
    margin_values = (q + margin * (torch.arange(q.shape[1], device=q.device)[None, :] != actions[:, None]))
    margin_values = margin_values.masked_fill(~action_mask, -torch.inf).max(-1).values - selected
    margin_values = margin_values.clamp_min(0)
    rows = row1 + row10 + demo.to(q.dtype) * margin_values
    params = list(trainable_parameters)
    regularizer = sum((p.square().sum() for p in params), q.new_zeros(()))
    loss = (weights.to(q.dtype) * rows).mean() + l2 * regularizer
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite DQfD objective")
    return loss, {"td1": d1.detach(), "td10": d10.detach(), "margin": margin_values.detach(),
                  "regularizer": regularizer.detach(), "selected_q": selected.detach()}


class GlobalPER:
    """One global proportional distribution; generations identify online rows."""
    def __init__(self, demo_count: int, online_capacity: int, alpha: float = ALPHA):
        if demo_count < 1 or online_capacity < 1:
            raise ValueError("positive replay capacities required")
        self.demo_count, self.online_capacity, self.alpha = demo_count, online_capacity, alpha
        self.demo_base = np.ones(demo_count, np.float64)
        self.online_base = np.zeros(online_capacity, np.float64)
        self.online_generation = np.full(online_capacity, -1, np.int64)
        self.online_count = 0

    def insert_online(self, slot: int, generation: int) -> None:
        if (not isinstance(slot, (int, np.integer)) or isinstance(slot, (bool, np.bool_))
                or not isinstance(generation, (int, np.integer)) or isinstance(generation, (bool, np.bool_))
                or not 0 <= slot < self.online_capacity or generation < 0):
            raise ValueError("invalid online slot/generation")
        if self.online_generation[slot] >= 0:
            raise ValueError("online slot must be evicted before insertion")
        active = self.online_base[(self.online_generation >= 0) & (np.arange(self.online_capacity) != slot)]
        # The frozen rule seeds online rows from CURRENT BASE priorities across
        # the whole replay. Demo bonus is only part of sampling mass.
        new_base = max(1.0, float(self.demo_base.max()), float(active.max()) if len(active) else 1.0)
        self.online_generation[slot] = generation
        self.online_base[slot] = new_base
        self.online_count = int(np.count_nonzero(self.online_generation >= 0))

    def probabilities(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        live = np.flatnonzero(self.online_generation >= 0)
        demo_total = self.demo_base + DEMO_BONUS
        masses = np.concatenate((demo_total, self.online_base[live])) ** self.alpha
        p = masses / masses.sum()
        return p, np.arange(self.demo_count), live

    def weights(self, probabilities: np.ndarray, beta: float) -> np.ndarray:
        w = (len(probabilities) * probabilities) ** (-beta)
        return w / w.max()

    def sample(self, count: int, rng: np.random.Generator, beta: float) -> dict[str, np.ndarray]:
        p, demo_ids, online_slots = self.probabilities()
        sampled = rng.choice(len(p), size=int(count), replace=True, p=p)
        demo = sampled < self.demo_count
        ids = sampled.copy()
        if np.any(~demo): ids[~demo] = online_slots[sampled[~demo] - self.demo_count]
        generations = np.full(len(sampled), -1, np.int64)
        if np.any(~demo): generations[~demo] = self.online_generation[ids[~demo]]
        weights = (len(p) * p[sampled]) ** (-beta)
        weights = weights / weights.max()
        return {"is_demo": demo, "ids": ids.astype(np.int64), "generation": generations.astype(np.int64),
                "probability": p[sampled], "weights": weights.astype(np.float32),
                "eligible_count": np.full(len(sampled), len(p), np.int64)}

    def update(self, ids: np.ndarray, generations: np.ndarray, td1: np.ndarray, td10: np.ndarray,
               is_demo: np.ndarray) -> None:
        # Duplicate IDs/generations aggregate max absolute error; epsilon/bonus once.
        arrays = [np.asarray(x) for x in (ids, generations, td1, td10, is_demo)]
        if any(x.ndim != 1 for x in arrays) or len({len(x) for x in arrays}) != 1:
            raise ValueError("PER update arrays must be equal-length vectors")
        if arrays[0].dtype.kind not in "iu" or arrays[1].dtype.kind not in "iu":
            raise ValueError("PER ids and generations must be integers")
        if arrays[4].dtype.kind != "b":
            raise ValueError("PER demo flags must be boolean")
        if not np.isfinite(arrays[2].astype(np.float64)).all() or not np.isfinite(arrays[3].astype(np.float64)).all():
            raise FloatingPointError("nonfinite TD error")
        maxima: dict[tuple[bool, int, int], float] = {}
        for idx, gen, a, b, demo in zip(*arrays):
            if not isinstance(demo, (bool, np.bool_)):
                raise ValueError("PER demo flag must be boolean")
            if not np.isfinite(float(a)) or not np.isfinite(float(b)):
                raise FloatingPointError("nonfinite TD error")
            key = (bool(demo), int(idx), int(gen))
            maxima[key] = max(maxima.get(key, 0.0), abs(float(a)), abs(float(b)))
        # Validate the complete update before changing any priority.
        for demo, idx, gen in maxima:
            if demo:
                if not 0 <= idx < self.demo_count or gen != -1: raise ValueError("stale/invalid demo id")
            elif not 0 <= idx < self.online_capacity or self.online_generation[idx] != gen:
                raise ValueError("stale online generation")
        for (demo, idx, gen), err in maxima.items():
            if demo:
                self.demo_base[idx] = err + PRIORITY_EPSILON
            else:
                self.online_base[idx] = err + PRIORITY_EPSILON

    def evict_online(self, slot: int, generation: int) -> None:
        if not 0 <= slot < self.online_capacity or int(self.online_generation[slot]) != int(generation):
            raise ValueError("stale online eviction generation")
        self.online_generation[slot] = -1
        self.online_base[slot] = 0.0
        self.online_count = int(np.count_nonzero(self.online_generation >= 0))


class OwnerNStep:
    """Independent owner queues; never joins episode/source transitions."""
    def __init__(self, gamma: float, n: int = N_STEP, max_owners: int = 6):
        self.gamma, self.n, self.max_owners = gamma, n, max_owners
        self.queues: dict[int, deque] = {}
        self.pending_dropped_at_end = 0

    def push(self, owner: int, row: dict[str, Any]) -> list[dict[str, Any]]:
        if owner not in self.queues and len(self.queues) >= self.max_owners:
            raise ValueError("owner capacity exceeded")
        queue = self.queues.setdefault(owner, deque())
        row = dict(row, owner=owner)
        queue.append(row)
        ready = []
        if row.get("boundary"):
            while queue:
                ready.append(paired_return(list(queue), self.gamma, self.n)); queue.popleft()
        elif len(queue) >= self.n:
            ready.append(paired_return(list(queue), self.gamma, self.n)); queue.popleft()
        return ready

    def finish(self) -> int:
        self.pending_dropped_at_end = sum(map(len, self.queues.values()))
        self.queues.clear()
        return self.pending_dropped_at_end


class LosslessFrameRing:
    """Small/reference-counted frame ring; callers pin every referenced slot.

    A reservation is atomic: all required free/evictable slots are found before
    writes begin. Generations make old references fail after slot reuse.
    """
    def __init__(self, capacity: int, frame_shapes: dict[str, tuple[int, ...]],
                 frame_dtypes: dict[str, Any]):
        if capacity < 1 or set(frame_shapes) != set(frame_dtypes):
            raise ValueError("invalid frame-ring schema")
        self.capacity = int(capacity)
        self.frames = {k: np.empty((capacity, *shape), dtype=frame_dtypes[k]) for k, shape in frame_shapes.items()}
        self.generation = np.full(capacity, -1, np.int64)
        self.refs = np.zeros(capacity, np.int32)
        self.cursor = 0; self.next_generation = 0; self.writes = 0
        self._content: dict[bytes, list[tuple[int, int]]] = {}
        self._slot_digest: list[bytes | None] = [None] * capacity

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.frames.values()) + self.generation.nbytes + self.refs.nbytes

    def reserve(self, count: int) -> list[tuple[int, int]]:
        if count < 0 or count > self.capacity: raise ValueError("reservation exceeds ring capacity")
        if count == 0: return []
        selected = []
        for offset in range(self.capacity):
            slot = (self.cursor + offset) % self.capacity
            if self.refs[slot] == 0:
                selected.append(slot)
                if len(selected) == count: break
        if len(selected) != count: raise RuntimeError("ring capacity exhausted by pinned references")
        return [(i, self.next_generation + n) for n, i in enumerate(selected)]

    def write(self, payload: dict[str, np.ndarray]) -> tuple[int, int]:
        if set(payload) != set(self.frames): raise ValueError("frame schema mismatch")
        converted = {}
        for key, frame in payload.items():
            raw = np.asarray(frame); cast = raw.astype(self.frames[key].dtype)
            if raw.shape != self.frames[key].shape[1:] or not np.array_equal(raw, cast):
                raise ValueError("lossy or shape-mismatched frame write")
            converted[key] = cast
        slot, generation = self.reserve(1)[0]
        prior_digest = self._slot_digest[slot]
        # Prepare all conversions and reserve before touching data/index state.
        try:
            for key, cast in converted.items(): self.frames[key][slot] = cast
        except BaseException:
            raise
        if prior_digest is not None:
            stale = (slot, int(self.generation[slot]))
            bucket = self._content.get(prior_digest, [])
            self._content[prior_digest] = [ref for ref in bucket if ref != stale]
            if not self._content[prior_digest]: self._content.pop(prior_digest, None)
            self._slot_digest[slot] = None
        self.generation[slot] = generation; self.cursor = (slot + 1) % self.capacity
        self.next_generation += 1; self.writes += 1
        return slot, generation

    @staticmethod
    def _digest_payload(payload: dict[str, np.ndarray]) -> bytes:
        h = hashlib.sha256()
        for key in sorted(payload):
            value = np.ascontiguousarray(payload[key])
            h.update(key.encode()); h.update(value.dtype.str.encode())
            h.update(repr(value.shape).encode()); h.update(value.tobytes(order="C"))
        return h.digest()

    def intern(self, payload: dict[str, np.ndarray]) -> tuple[int, int]:
        """Pin an exact frame payload, deduplicating only after hash and full compare."""
        if set(payload) != set(self.frames): raise ValueError("frame schema mismatch")
        converted = {}
        for key in sorted(payload):
            raw = np.asarray(payload[key]); cast = raw.astype(self.frames[key].dtype)
            if raw.shape != self.frames[key].shape[1:] or not np.array_equal(raw, cast):
                raise ValueError("lossy or shape-mismatched frame write")
            converted[key] = cast
        digest = self._digest_payload(converted)
        candidates = self._content.get(digest, [])
        for slot, generation in candidates:
            if self.generation[slot] == generation and all(np.array_equal(self.frames[key][slot], converted[key]) for key in converted):
                self.pin([(slot, generation)])
                return slot, generation
        reference = self.write(converted)
        self.pin([reference]); self._content.setdefault(digest, []).append(reference); self._slot_digest[reference[0]] = digest
        return reference

    def pin(self, refs: Iterable[tuple[int, int]]) -> None:
        refs = list(refs)
        counts: dict[tuple[int, int], int] = {}
        for slot, generation in refs:
            if not 0 <= slot < self.capacity or self.generation[slot] != generation:
                raise ValueError("stale frame generation")
            key = (int(slot), int(generation)); counts[key] = counts.get(key, 0) + 1
        for (slot, _), count in counts.items():
            if int(self.refs[slot]) + count > np.iinfo(np.int32).max:
                raise OverflowError("frame reference count overflow")
        for (slot, _), count in counts.items(): self.refs[slot] += count

    def unpin(self, refs: Iterable[tuple[int, int]]) -> None:
        refs = list(refs)
        counts: dict[tuple[int, int], int] = {}
        for slot, generation in refs:
            if not 0 <= slot < self.capacity or self.generation[slot] != generation:
                raise ValueError("stale or unbalanced frame release")
            key = (int(slot), int(generation)); counts[key] = counts.get(key, 0) + 1
        for (slot, _), count in counts.items():
            if int(self.refs[slot]) < count: raise ValueError("stale or unbalanced frame release")
        for (slot, _), count in counts.items(): self.refs[slot] -= count

    def gather(self, refs: Iterable[tuple[int, int]]) -> dict[str, np.ndarray]:
        refs = list(refs)
        for slot, generation in refs:
            if not 0 <= slot < self.capacity or self.generation[slot] != generation:
                raise ValueError("stale frame generation")
        return {key: np.stack([array[slot].copy() for slot, _ in refs]) for key, array in self.frames.items()}


def build_q_from_actor(actor: nn.Module, shapes: dict[str, tuple[int, ...]], actions: int,
                       config: dict[str, Any], seed: int, device: str = "cpu") -> tuple[DuelingQ, DuelingQ]:
    torch.manual_seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    online = DuelingQ(shapes, actions, config)
    online.transfer_encoder(actor)
    target = copy.deepcopy(online).to(device).eval()
    online = online.to(device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    return online, target


def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8, betas=(.9, .999), weight_decay=0.)


def optimizer_step(model: nn.Module, target: nn.Module, optimizer: torch.optim.Optimizer,
                   loss: torch.Tensor, step: int, clip: float = 10., *,
                   applied_callback=None, target_copy_callback=None) -> tuple[int, float, bool]:
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
    optimizer.zero_grad(set_to_none=True); loss.backward()
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError("nonfinite gradient")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
    if not math.isfinite(norm): raise FloatingPointError("nonfinite gradient norm")
    optimizer.step()
    step += 1
    if applied_callback is not None:
        applied_callback(step)
    for p in model.parameters():
        if not torch.isfinite(p).all(): raise FloatingPointError("nonfinite parameter")
    synced = step % 1000 == 0
    if synced:
        target.load_state_dict(model.state_dict()); target.eval()
        if target_copy_callback is not None:
            target_copy_callback(step)
    return step, norm, synced


def output_comparison(reference: torch.Tensor, actual: torch.Tensor, *, atol: float = 1e-5,
                      rtol: float = 1e-5) -> dict[str, Any]:
    """Report numerical output agreement separately from serialized-state hashes."""
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError("Q-output comparison requires matching batch-by-action tensors")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    if finite:
        delta = (reference - actual).abs()
        max_abs = float(delta.max())
        close = bool(torch.allclose(reference, actual, atol=atol, rtol=rtol))
        same_argmax = torch.equal(reference.argmax(-1), actual.argmax(-1))
    else:
        max_abs, close, same_argmax = float("inf"), False, False
    return {"finite": finite, "max_abs_error": max_abs, "atol": float(atol),
            "rtol": float(rtol), "within_tolerance": close,
            "same_action_ranking": bool(same_argmax),
            "reference_argmax": reference.argmax(-1).detach().cpu().tolist() if finite else [],
            "actual_argmax": actual.argmax(-1).detach().cpu().tolist() if finite else []}


class OnlineCadence:
    """Deterministic aggregate-frame/update ledger; performs no environment work."""
    def __init__(self, cap: int = 368640, warmup: int = 4096, cadence: int = 64, workers: int = 6):
        if cap <= warmup or warmup < 0 or cadence < 1 or workers < 1:
            raise ValueError("invalid online frame schedule")
        self.cap, self.warmup, self.cadence, self.workers = cap, warmup, cadence, workers
        self.frames = 0; self.events = 0; self.next_event = warmup + cadence; self.rounds = 0

    def dispatch_width(self) -> int:
        remaining = self.cap - self.frames
        return self.workers if remaining >= self.workers else 0

    def complete_round(self, completed_actions: int) -> int:
        width = self.dispatch_width()
        if width == 0 or completed_actions != width:
            raise ValueError("collector round does not fit the aggregate frame cap")
        before = self.frames; self.frames += completed_actions; self.rounds += 1
        credited = 0
        while self.frames >= self.next_event and self.next_event > before:
            credited += 1; self.events += 1; self.next_event += self.cadence
        return credited


class GreedyQPolicy:
    """Inference-only adapter: direct Q argmax, no actor softmax/temperature."""
    def __init__(self, model: nn.Module):
        self.model = model.eval(); self.device = next(model.parameters()).device; self.calls = 0

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray]) -> int:
        q = self.model(window_tensors(observation, self.device))
        if q.ndim != 2 or q.shape[0] != 1 or not torch.isfinite(q).all():
            raise FloatingPointError("invalid greedy Q output")
        self.calls += 1
        return int(q.argmax(-1).item())


def evaluate_greedy_episode(policy: GreedyQPolicy, reset_fn, step_fn, frame_cap: int) -> dict[str, Any]:
    """Injected endpoint adapter; never creates an environment or fabricates a terminal."""
    if frame_cap < 1: raise ValueError("positive gameplay frame cap required")
    observation, reset_receipt = reset_fn()
    frames, ended, outcome, last = 0, False, None, None
    while frames < frame_cap:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = step_fn(action)
        frames += int(info.get("native_frames", 1)); last = info
        if frames > frame_cap: raise RuntimeError("endpoint exceeded native-frame cap")
        if terminated or truncated:
            ended, outcome = True, info.get("end_reason", "terminated" if terminated else "time_limit")
            break
    return {"frames": frames, "ended": ended, "end_reason": outcome,
            "unfinished_at_cap": not ended, "reset_receipt": reset_receipt,
            "terminal_receipt": last if ended else None, "inference_calls": policy.calls}
