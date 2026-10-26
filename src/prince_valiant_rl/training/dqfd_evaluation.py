"""Serial reset-389 Q evaluation and same-episode video encoding."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import numpy as np


def evaluate_reset389_video(policy: Any, env_factory: Callable[[], Any], *,
                            identity_receipt: Callable[[Any], Mapping[str, Any]],
                            reset_verifier: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
                            output_path: Path, receipt_fn: Callable[[dict[str, Any]], Any],
                            frame_cap: int = 36000, record_every: int = 2,
                            checkpoint_update: int | None = None,
                            checkpoint_sha256: str | None = None,
                            encoder_factory: Callable[..., Any] | None = None,
                            frame_reader: Callable[[Any], np.ndarray] | None = None,
                            video_byte_cap: int = 100 * 1024 * 1024) -> dict[str, Any]:
    """Evaluate one real Level1Env episode; all native-dependent imports are lazy/injected.

    ``identity_receipt`` must persist runtime identity before reset. The reset
    verifier checks the trusted route/snapshot lineage and zero restoration
    advances. Screenshots are read from the same native owner while it runs.
    """
    if not 1 <= int(frame_cap) <= 36000 or record_every < 1:
        raise ValueError("reset-389 horizon/cadence exceeds the frozen bound")
    if not callable(receipt_fn): raise ValueError("append-only episode receipt callback is required")
    if encoder_factory is None:
        import imageio_ffmpeg
        encoder_factory = imageio_ffmpeg.write_frames
    path = Path(output_path)
    if path.exists(): raise FileExistsError(f"video destination is occupied: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    if partial.exists(): raise FileExistsError(f"partial video destination is occupied: {partial}")
    env = env_factory()
    writer = None
    result: dict[str, Any] = {"status": "started", "checkpoint_update": None,
        "condition": "greedy_reset389", "reset_frame": 389, "epsilon": 0.0,
        "frame_cap": int(frame_cap), "record_every_native_frames": int(record_every),
        "native_frames": 0, "recorded_frames": 0, "raw_frame_bytes": 0,
        "native_step_attempted": 0, "native_step_completed": 0,
        "native_step_uncertain": 0, "native_frames_uncertain": 0,
        "video_bytes": 0, "score_availability": None, "frame_timestamp_count": 1,
        "frame_timestamp_sha256": hashlib.sha256(b"0").hexdigest(),
        "first_frame_native_timestamp": 0, "last_frame_native_timestamp": 0}
    result["checkpoint_update"] = checkpoint_update
    result["checkpoint_sha256"] = checkpoint_sha256
    receipt_written = False
    receipt_attempted = False
    try:
        result["runtime_identity"] = dict(identity_receipt(env))
        result["runtime_identity_matched"] = (result["runtime_identity"].get("verified") is True
            and result["runtime_identity"].get("persisted_before_reset") is True)
        if not result["runtime_identity_matched"]:
            raise RuntimeError("runtime identity receipt was not durably verified before reset")
        if frame_reader is None: frame_reader = lambda owner: owner.native.copy_screen_rgb()
        observation, reset_info = env.reset(389, horizon=int(frame_cap))
        verifier_params = inspect.signature(reset_verifier).parameters
        if len(verifier_params) >= 3 or any(p.kind == p.VAR_POSITIONAL for p in verifier_params.values()):
            provenance = dict(reset_verifier(env, reset_info, observation))
        else:
            provenance = dict(reset_verifier(env, reset_info))
        result["reset_provenance"] = provenance
        result["reset_provenance_verified"] = provenance.get("verified") is True
        if provenance.get("verified") is not True or int(provenance.get("restoration_native_frames", -1)) != 0:
            raise RuntimeError("reset-389 snapshot provenance/zero-advance verification failed")
        initial_objectives = set(getattr(getattr(env, "tracker", None), "objectives", ()))
        screen = np.ascontiguousarray(frame_reader(env), dtype=np.uint8)
        if screen.shape != (240, 256, 3): raise ValueError("production screenshot must be 256x240 RGB")
        result["raw_frame_bytes"] += int(screen.nbytes)
        writer = encoder_factory(str(partial), (256, 240), fps=30, codec="libx264",
            pix_fmt_in="rgb24", output_params=["-crf", "24"], ffmpeg_log_level="error")
        writer.send(None); writer.send(screen)
        result["recorded_frames"] = 1
        reward_total = 0.0; native_frames = 0; terminal = truncated = False
        action_count = 0; last_info: dict[str, Any] = {}; trace = hashlib.sha256()
        action_catalog = tuple(map(int, env.catalog))
        if len(action_catalog) != 17 or len(set(action_catalog)) != 17:
            raise ValueError("production action catalog must contain 17 distinct raw masks")
        while not (terminal or truncated):
            action = int(policy.act(observation)); action_count += 1
            if not 0 <= action < 17: raise ValueError("Q action is outside the frozen 17-action mapping")
            # Count the native boundary before calling it. If the owner advances
            # and then raises while assembling reward/info, the receipt must not
            # imply that no native work occurred.
            result["native_step_attempted"] += 1
            try:
                step_result = env.step(action)
            except BaseException:
                result["native_step_uncertain"] += 1
                result["native_frames_uncertain"] += 1
                result["native_frames_lower_bound"] = native_frames
                result["native_frames"] = None
                raise
            result["native_step_completed"] += 1
            try:
                observation, reward, terminal, truncated, last_info = step_result
                if not isinstance(last_info, dict) or not isinstance(last_info.get("availability", {}), dict):
                    raise ValueError("native step returned malformed info/availability")
                duration = int(last_info.get("transition_frames", 0))
                reward_value = float(reward)
                if not math.isfinite(reward_value):
                    raise ValueError("native step returned nonfinite reward")
            except BaseException:
                result["native_step_uncertain"] += 1
                result["native_frames_uncertain"] += 1
                # A completed native call with malformed info/reward does not
                # establish the amount of native time advanced. Keep the
                # already verified total as a lower bound, and leave the total
                # unknown instead of retaining the initialized zero.
                result["native_frames_lower_bound"] = native_frames
                result["native_frames"] = None
                raise
            if duration < 1 or native_frames + duration > frame_cap:
                result["native_step_uncertain"] += 1
                result["native_frames_uncertain"] += max(1, duration)
                result["native_frames_lower_bound"] = native_frames
                result["native_frames"] = None
                raise RuntimeError("actual native duration is invalid or exceeds the cap")
            native_frames += duration; reward_total += reward_value
            trace.update(json.dumps({"action": action, "raw_mask": action_catalog[action],
                "duration": duration}, sort_keys=True, separators=(",", ":")).encode())
            result["native_frames"] = native_frames
            if native_frames % record_every == 0 or terminal or truncated:
                screen = np.ascontiguousarray(frame_reader(env), dtype=np.uint8)
                if screen.shape != (240, 256, 3): raise ValueError("production screenshot must be 256x240 RGB")
                result["raw_frame_bytes"] += int(screen.nbytes)
                result["recorded_frames"] += 1
                # Keep a bounded digest of the exact same-episode capture cadence.
                timestamp = str(native_frames).encode("ascii")
                result["frame_timestamp_sha256"] = hashlib.sha256(
                    bytes.fromhex(result["frame_timestamp_sha256"]) + timestamp).hexdigest()
                result["frame_timestamp_count"] += 1
                result["last_frame_native_timestamp"] = native_frames
                writer.send(screen)
                if partial.exists() and partial.stat().st_size > video_byte_cap:
                    raise RuntimeError("encoded video exceeded the 100 MiB per-bundle cap")
            if native_frames == frame_cap and not (terminal or truncated): break
        result.update(status="episode_captured", action_count=action_count,
            cumulative_applied_reward=reward_total, end_reason=last_info.get("end_reason") or
                ("time_limit" if truncated else "frame_cap" if native_frames == frame_cap else "unknown"),
            frame_cap_reached=bool(native_frames == frame_cap and not (terminal or truncated)),
            natural_terminal=bool(terminal and last_info.get("game_terminal")),
            terminal_valid=bool(terminal or truncated),
            score_availability=last_info.get("availability", {}).get("score_gain"),
            verified_objective_clear_identities=sorted(set(getattr(
                getattr(env, "tracker", None), "objectives", ())) - initial_objectives),
            final_state_captured_before_reset=True, action_catalog=list(action_catalog),
            action_trace_sha256=trace.hexdigest(), video_partial_path=str(partial),
            raw_frame_live_bytes_max=240 * 256 * 3)
        writer.close(); writer = None
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise RuntimeError("video encoder produced no verifiable output")
        if partial.stat().st_size > video_byte_cap:
            raise RuntimeError("encoded video exceeded the 100 MiB per-bundle cap")
        os.replace(partial, path)
        result.update(status="complete", video_path=str(path), video_bytes=path.stat().st_size,
            video_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return result
    except BaseException as exc:
        result.update(status="incomplete", error=repr(exc)[:1000],
            video_partial_path=str(partial) if partial.exists() else str(path) if path.exists() else None,
            video_bytes=partial.stat().st_size if partial.exists() else path.stat().st_size if path.exists() else 0)
        try:
            receipt_attempted = True
            receipt_fn(result); receipt_written = True
        except Exception as receipt_exc: exc.add_note(f"failure receipt also failed: {receipt_exc!r}")
        raise
    finally:
        active_error = sys.exc_info()[1]
        errors = []
        if writer is not None:
            try: writer.close()
            except Exception as exc: errors.append(exc)
        try: env.close()
        except Exception as exc: errors.append(exc)
        if errors:
            result.update(status="incomplete", cleanup_errors=[repr(exc)[:300] for exc in errors])
            try:
                receipt_attempted = True
                receipt_fn(result); receipt_written = True
            except Exception as receipt_exc: errors.append(receipt_exc)
            error = RuntimeError("evaluation child/owner cleanup failed: " + "; ".join(repr(e)[:200] for e in errors))
            active = sys.exc_info()[1]
            if active is not None: active.add_note(str(error))
            else: raise error
        elif not receipt_written and not receipt_attempted:
            try:
                receipt_attempted = True
                receipt_fn(result); receipt_written = True
            except Exception as receipt_exc:
                if active_error is not None:
                    active_error.add_note(f"episode receipt also failed: {receipt_exc!r}")
                else:
                    raise
