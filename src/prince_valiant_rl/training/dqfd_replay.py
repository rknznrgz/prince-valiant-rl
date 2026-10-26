"""Bounded demonstration gather over the integrity-checked v4 cache reader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DemoRow:
    global_id: int
    source: str
    source_id: int
    local_row: int
    reward1: float
    duration1: int
    discount1: float
    bootstrap1: bool
    reward10: float
    duration10: int
    discount10: float
    bootstrap10: bool
    following1: int
    following10: int
    raw_mask: int
    action: int
    valid: bool


class DemoReplayIndex:
    """Scalar index is built once; window tensors are gathered only on demand.

    ``cache`` must be an open production ``TransitionCache``. Its READY,
    manifest, file hashes, references, identity and schema have already passed
    that reader's normal validation. The index never edits or owns cache arrays.
    """

    def __init__(self, cache: Any, gamma: float = .99997, n_step: int = 10):
        if not 0 < gamma <= 1 or n_step != 10:
            raise ValueError("frozen DQfD gamma and ten-step horizon required")
        self.cache, self.gamma, self.n_step = cache, float(gamma), int(n_step)
        self.rows: list[DemoRow] = []
        self.by_source: dict[str, list[int]] = {}
        self._priority = np.ones(int(cache.manifest["rows"]), dtype=np.float64)
        expected = {item["name"]: int(item["rows"]) for item in cache.manifest["sources"]}
        source_ids = {item["name"]: int(item["source_id"]) for item in cache.manifest["sources"]}
        local: dict[tuple[str, int], dict[str, Any]] = {}
        global_keys: list[tuple[str, int]] = []
        for record in cache:
            source, row = str(record["trajectory"]), int(record["local_row"])
            if source not in expected or int(record["source_id"]) != source_ids[source]:
                raise ValueError("wrong-source or unaccepted demo reference")
            if row != len(self.by_source.setdefault(source, [])):
                raise ValueError("noncontiguous source-local demonstration rows")
            ref = record["next_ref"]
            scalar = {"trajectory": source, "source_id": source_ids[source], "local_row": row,
                "reward": float(record["reward"]), "duration": int(record["duration"]),
                "bootstrap": bool(record["bootstrap"]), "natural_terminal": bool(record["natural_terminal"]),
                "time_limit": bool(record["time_limit"]), "action": int(record["action"]),
                "raw_mask": int(record["raw_mask"]),
                "next_ref": {"kind": str(ref["kind"]), "row": int(ref["row"]),
                             **({"trajectory": str(ref["trajectory"])} if "trajectory" in ref else {})}}
            global_id = len(global_keys)
            local[(source, row)] = scalar
            global_keys.append((source, row)); self.by_source[source].append(global_id)
        if {k: len(v) for k, v in self.by_source.items()} != expected:
            raise ValueError("accepted source partition differs from frozen counts")
        if len(global_keys) != sum(expected.values()):
            raise ValueError("accepted receipt count differs from manifest")
        self.rows = [None] * len(global_keys)
        for global_id, (source, row) in enumerate(global_keys):
            record = local[(source, row)]
            refs = self.by_source[source]
            rr, disc, duration, last = 0.0, 1.0, 0, record
            for local_row in range(row, min(row + n_step, len(refs))):
                current = local[(source, local_row)]
                if current["trajectory"] != source or int(current["local_row"]) != local_row:
                    raise ValueError("n-step tail crossed source boundary")
                rr += disc * float(current["reward"])
                duration += int(current["duration"])
                disc *= gamma ** int(current["duration"])
                last = current
                if current["next_ref"]["kind"] == "final_post":
                    break
            first_ref = record["next_ref"]
            if first_ref.get("kind") == "pre":
                if first_ref.get("trajectory", source) != source:
                    raise ValueError("wrong-source one-step target reference")
                first_local = int(first_ref["row"])
                if first_local != row + 1 or not 0 <= first_local < len(refs):
                    raise ValueError("stale/corrupt one-step target row")
                following1 = refs[first_local]
            else:
                if first_ref.get("trajectory") != source:
                    raise ValueError("wrong-source one-step final reference")
                if int(first_ref.get("row", -1)) != row:
                    raise ValueError("stale/corrupt final one-step reference")
                following1 = -1  # final_post marker; resolved by reader at gather time
            last_ref = last["next_ref"]
            if last_ref.get("kind") == "pre":
                if last_ref.get("trajectory", source) != source:
                    raise ValueError("wrong-source n-step target reference")
                last_local = int(last_ref["row"])
                if not 0 <= last_local < len(refs): raise ValueError("stale/corrupt n-step target row")
                following10 = refs[last_local]
            else:
                if last_ref.get("trajectory") != source:
                    raise ValueError("wrong-source n-step final reference")
                if int(last_ref.get("row", -1)) != int(last["local_row"]):
                    raise ValueError("stale/corrupt final n-step reference")
                following10 = -1
            self.rows[global_id] = DemoRow(global_id, source, int(record["source_id"]), row,
                float(record["reward"]), int(record["duration"]), gamma ** int(record["duration"]),
                bool(record["bootstrap"]), rr, duration, disc, bool(last["bootstrap"]),
                following1, following10, int(record["raw_mask"]), int(record["action"]), True)
        self._records = local

    def update_priority(self, ids: Sequence[int], values: Sequence[float]) -> None:
        idx, val = np.asarray(ids, dtype=np.int64), np.asarray(values, dtype=np.float64)
        if idx.ndim != 1 or val.shape != idx.shape or np.any(idx < 0) or np.any(idx >= len(self.rows)):
            raise ValueError("invalid demo priority update")
        if not np.isfinite(val).all() or np.any(val < 0):
            raise ValueError("invalid demo priority")
        aggregated: dict[int, float] = {}
        for row_id, priority in zip(idx, val):
            aggregated[int(row_id)] = max(aggregated.get(int(row_id), 0.0), float(priority))
        for row_id, priority in aggregated.items(): self._priority[row_id] = priority

    def gather(self, ids: Sequence[int], *, probabilities: Sequence[float] | None = None,
               weights: Sequence[float] | None = None) -> dict[str, Any]:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) < 1 or len(ids) > 128 or np.any(ids < 0) or np.any(ids >= len(self.rows)):
            raise ValueError("demo gather requires 1..128 valid global IDs")
        probs = np.ones(len(ids), np.float64) / len(self.rows) if probabilities is None else np.asarray(probabilities, np.float64)
        isw = np.ones(len(ids), np.float32) if weights is None else np.asarray(weights, np.float32)
        if probs.shape != ids.shape or isw.shape != ids.shape or not np.isfinite(probs).all() or np.any(probs <= 0) or not np.isfinite(isw).all() or np.any(isw < 0):
            raise ValueError("invalid sample probability/weight metadata")
        selected = [self.rows[int(i)] for i in ids]
        current, one, ten = [], [], []
        for row in selected:
            current.append(self.cache.window(row.source, "pre", row.local_row))
            one.append(self.cache.window(row.source, "final_post" if row.following1 < 0 else "pre",
                                         0 if row.following1 < 0 else self.rows[row.following1].local_row))
            ten.append(self.cache.window(row.source, "final_post" if row.following10 < 0 else "pre",
                                         0 if row.following10 < 0 else self.rows[row.following10].local_row))
        def stack(items):
            keys = set(items[0])
            if any(set(item) != keys for item in items): raise ValueError("window field schema mismatch")
            return {key: np.stack([item[key] for item in items]) for key in items[0]}
        return {"current": stack(current), "following1": stack(one), "following10": stack(ten),
            "actions": np.asarray([r.action for r in selected], np.int64),
            "action_masks": np.ones((len(ids), 17), np.bool_),
            "returns1": np.asarray([r.reward1 for r in selected], np.float64),
            "durations1": np.asarray([r.duration1 for r in selected], np.int64),
            "discounts1": np.asarray([r.discount1 for r in selected], np.float64),
            "bootstrap1": np.asarray([r.bootstrap1 for r in selected], np.bool_),
            "returns10": np.asarray([r.reward10 for r in selected], np.float64),
            "durations10": np.asarray([r.duration10 for r in selected], np.int64),
            "discounts10": np.asarray([r.discount10 for r in selected], np.float64),
            "bootstrap10": np.asarray([r.bootstrap10 for r in selected], np.bool_),
            "demo_ids": ids.copy(), "source_ids": np.asarray([r.source_id for r in selected], np.int64),
            "is_demo": np.ones(len(ids), np.bool_),
            "sources": [r.source for r in selected], "probabilities": probs,
            "weights": isw, "valid": np.ones(len(ids), np.bool_),
            "measurement": {"requested_rows": int(len(ids)), "window_reads": int(3 * len(ids)),
                            "window_roles": ["pre", "following1", "following10"]},
            "history_valid": np.stack([current[i]["history_valid"] for i in range(len(ids))]),
            "raw_masks": np.asarray([r.raw_mask for r in selected], np.int64)}
