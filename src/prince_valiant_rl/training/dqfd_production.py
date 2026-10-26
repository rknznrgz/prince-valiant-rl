"""Pinned production factories for the demo-only DQfD pretraining slice.

Imports are safe for help/preflight. Cache, actor, Q models and optimizer are
constructed only when ``create_fresh_demo_learner`` is explicitly called by a
supervised child.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import secrets
import hmac
import os
import time
import uuid

from .dqfd import build_q_from_actor, optimizer_for
from .dqfd_execution import DQfDBatchLearner
from .dqfd_replay import DemoReplayIndex
from .dqfd_transitions import TransitionCache, sha256_file
from .temporal_models import WindowPath
from .dqfd_pretraining import load_committed_milestone
from .dqfd_evaluation import evaluate_reset389_video
from .dqfd_receipts import append_record

CACHE_REL = "training/level1_dqfd_transitions_reconstruction_v4_20261024"
ACTOR_REL = "analysis/level1_full_route_verification_20261023/epoch400_eval_actor.pt"
CONFIG_REL = "training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json"
PINS = {
    "manifest": "23c3b215462f2bf921a965a9e221e45e49447aafbb3b209da5dc30c9ee8d0a84",
    "READY": "bdc6e0c7f3688c56c255ab554479dc5de6a3ab39617cf7dfc4ca7857dcd00f8e",
    "identity": "a7590cafce66d89050638bdb44d63dd2767dad18622dbd0e376e6d9d0ce1683b",
    "actor": "780269fac004b733b3d8b002e8d4c9854f83958e83eeddce0d72b41e7fcab18e",
    "config": "f2a28db6efc046b8527fab61b39a21fff2de8e3762eedc4638f0eb8e193f6d7d",
}
SOURCE_ROWS = {"original_389": 18273, "human18000_noop2": 836,
               "human17940_noop2": 837, "ppo_noop1": 838, "ppo_noop16": 815}
ORIGINAL_389_PRE_WINDOW_SHA256 = "3f01f94115ab774faea29bc9f0510f1f7459b90c881c8665815831bbe8ae7a14"


def _window_digest(window: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(window):
        value = np.asarray(window[key])
        digest.update(key.encode("utf-8") + b"\0" + value.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _pinned_original_389_pre_window(root: Path) -> dict[str, np.ndarray]:
    """Read the immutable v4 PRE oracle through bounded, read-only mmap access."""
    from .dqfd_transitions import _FrameIndexedReader
    cache = Path(root) / CACHE_REL
    ready = json.loads((cache / "READY").read_text(encoding="utf-8"))
    manifest_path = cache / "manifest.json"
    manifest_hash = _sha256(manifest_path)
    if (ready.get("manifest_sha256") != PINS["manifest"]
            or manifest_hash != PINS["manifest"]
            or _sha256(cache / "READY") != PINS["READY"]):
        raise RuntimeError("pinned v4 reference manifest/READY identity drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next((row for row in manifest.get("sources", []) if row.get("name") == "original_389"), None)
    if (source is None or int(source.get("reset_frame", -1)) != 389
            or int(source.get("rows", -1)) != SOURCE_ROWS["original_389"]
            or tuple(manifest.get("history_offsets", ())) != (-60, -30, -15, -9, -6, -3, 0)):
        raise RuntimeError("pinned v4 original_389 source contract drift")
    reader = _FrameIndexedReader(cache, manifest)
    try:
        window = reader.window("original_389", "pre", 0)
    finally:
        reader.close()
    if _window_digest(window) != ORIGINAL_389_PRE_WINDOW_SHA256:
        raise RuntimeError("pinned v4 original_389 first PRE window differs from independent oracle")
    return window


def _same_array_exact(left: Any, right: Any) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes(order="C") == b.tobytes(order="C")


def _same_window_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (isinstance(left, dict) and set(left) == set(right)
            and all(_same_array_exact(left[key], right[key]) for key in right))


def production_reset389_episode(*, root: Path, output_dir: Path, receipt_path: Path,
        attempt_id: str, update: int, checkpoint_path: Path, checkpoint_sha256: str,
        label: str = "candidate",
        pins: dict[str, str], env_factory: Callable[[], Any] | None = None,
        identity_verifier: Callable[..., dict[str, Any]] | None = None,
        reset_verifier: Callable[..., dict[str, Any]] | None = None,
        screen_reader: Callable[[Any], Any] | None = None,
        video_writer: Callable[..., Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
        checkpoint_loader: Callable[..., Any] | None = None,
        inference_device: str = "cuda:0",
        receipt_fn: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Isolate checkpoint loading/inference RNG from the live learner RNG."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return _production_reset389_episode(root=root, output_dir=output_dir,
            receipt_path=receipt_path, attempt_id=attempt_id, update=update,
            label=label,
            checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha256,
            pins=pins, env_factory=env_factory, identity_verifier=identity_verifier,
            reset_verifier=reset_verifier, screen_reader=screen_reader,
            video_writer=video_writer, model_factory=model_factory,
            checkpoint_loader=checkpoint_loader, inference_device=inference_device,
            receipt_fn=receipt_fn)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)


def _production_reset389_episode(*, root: Path, output_dir: Path, receipt_path: Path,
        attempt_id: str, update: int, checkpoint_path: Path, checkpoint_sha256: str,
        label: str = "candidate",
        pins: dict[str, str], env_factory: Callable[[], Any] | None = None,
        identity_verifier: Callable[..., dict[str, Any]] | None = None,
        reset_verifier: Callable[..., dict[str, Any]] | None = None,
        screen_reader: Callable[[Any], Any] | None = None,
        video_writer: Callable[..., Any] | None = None,
        model_factory: Callable[..., Any] | None = None,
        checkpoint_loader: Callable[..., Any] | None = None,
        inference_device: str = "cuda:0",
        receipt_fn: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Production milestone callback; environment/native construction is lazy."""
    from .dqfd_pretraining import load_committed_milestone
    import json
    root = Path(root).resolve(); output_dir = Path(output_dir); receipt_path = Path(receipt_path)
    # Verify complete file identity before Q or native construction.
    model, checkpoint = (checkpoint_loader or load_committed_milestone)(checkpoint_path, expected_update=update,
        expected_pins=pins, expected_config_hash=PINS["config"], model_factory=model_factory)
    if checkpoint["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("milestone checkpoint hash differs from committed callback argument")
    model.to(inference_device)
    model.eval()
    class GreedyPolicy:
        def act(self, observation):
            from .temporal_models import window_tensors
            with torch.no_grad():
                tensors = window_tensors(observation, next(model.parameters()).device)
                return int(model(tensors).argmax(dim=1)[0].item())
    policy = GreedyPolicy()
    sequence = 0
    def durable(row):
        nonlocal sequence
        if receipt_fn is not None:
            receipt_fn(row)
        else:
            sequence += 1
            append_record(receipt_path, row, attempt_id=attempt_id, sequence=sequence)
    if env_factory is None:
        def env_factory():
            from prince_valiant_rl.backward_level1 import HumanRoute, Level1Env
            config_path = root / CONFIG_REL
            config = json.loads(config_path.read_text(encoding="utf-8"))
            route = HumanRoute(directory=root / "training/backward_level1_v1", config=config)
            return Level1Env(config, route=route)
    def identity(env):
        if identity_verifier is not None:
            return identity_verifier(env, root=root, pins=pins, receipt_fn=durable)
        route = env.route
        native = env.native
        abi = getattr(native, "abi_identity", None)
        if not isinstance(abi, dict):
            raise RuntimeError("configured native observer/base lineage is unavailable")
        route_sha = str(route.meta.get("route_sha256", ""))
        actual = {"config_sha256": _sha256(root / CONFIG_REL),
            "rom_sha256": str(native.rom_hash), "native_build_id": str(native.build),
            "route_sha256": route_sha,
            "route_metadata_sha256": _sha256(root / "training/backward_level1_v1/route.json"),
            "reset389_snapshot_sha256": _sha256(root / "training/backward_level1_v1/states/0000389.state.z"),
            "observer_abi_identity": abi,
            "observer_provenance_sha256": _sha256(Path(abi["provenance_path"])),
            "observer_dll_sha256": _sha256(Path(abi["observer_dll_path"])),
            "base_dll_sha256": _sha256(root / "instrumentation/recorder_build/recorder_native.dll")}
        expected = {"config_sha256": PINS["config"],
            "rom_sha256": "96ad0a41a34fccc98bb7255890b131f77d4131cc69594c77e3ff33cd3a8fe6a5",
            "route_sha256": "214b5d56f858a45690779c11441ee3251880a1e8bf3418e564f461fe655b6eed",
            "route_metadata_sha256": "d4b0cb74e06705671bca8699d692b410d7c06901a8e2098596007d3d918b26d6",
            "reset389_snapshot_sha256": "49bbaebd343ad4b19133d7b7788bef9d8868911224eb722a667b6a55fe7c4810",
            "native_build_id": "b45adda7fea78daaed5b9ae0b02255f8bac8b3cdf07d317a93134a261ae1ddca",
            "observer_provenance_sha256": "2095daea0727244383f7292d564ea855e539296c647cfcfee4c7169b8a5f058f",
            "observer_dll_sha256": "cb085620da457e09cda88a7638021ce23b1fc2b186561170c7cce90b0cebbaaa",
            "base_dll_sha256": "76bb13f0d723ca553d69445b435a1e5a475c4fe9264fb55ae42862eebbaa21ec"}
        if any(actual.get(key) != value for key, value in expected.items()):
            raise RuntimeError("runtime ROM/route/config identity drift")
        if (actual["observer_dll_sha256"] != abi.get("observer_dll_sha256")
                or actual["base_dll_sha256"] != abi.get("base_dll_sha256")
                or abi.get("abi") != 2
                or abi.get("base_build_id") != expected["native_build_id"]
                or actual["observer_provenance_sha256"] != expected["observer_provenance_sha256"]):
            raise RuntimeError("loaded observer/base native lineage differs from loaded-owner identity")
        receipt = {**actual, "verified": True, "persisted_before_reset": True}
        durable({"kind": "runtime_identity", **receipt})
        return receipt
    def reset(env, info, observation=None):
        if reset_verifier is not None:
            return reset_verifier(env, info, observation=observation, receipt_fn=durable)
        if int(info.get("reset_frame", -1)) != 389 or int(info.get("reset_replay_frames", -1)) != 0:
            raise RuntimeError("reset-389 route restoration did not match trusted zero-advance evidence")
        route = env.route
        if int(route.start) != 389:
            raise RuntimeError("reset-389 HumanRoute start differs from trusted route-start contract")
        if not _same_array_exact(env.ram, route.ram[389]):
            raise RuntimeError("reset-389 RAM differs from verified HumanRoute state")
        if int(env.last_action) != int(route.buttons[388]):
            raise RuntimeError("reset-389 controller state differs from HumanRoute history")
        if not isinstance(observation, dict):
            raise RuntimeError("reset-389 observation history is malformed")
        actual_window = env.history.window()
        if not _same_window_exact(observation, actual_window):
            raise RuntimeError("reset-389 returned policy input differs from owner history")
        offsets = tuple(int(x) for x in env.history.offsets)
        earliest = max(route.start, 389 + min(offsets))
        expected_current_index = 389 - earliest
        if (offsets != (-60, -30, -15, -9, -6, -3, 0)
                or int(env.history.current) != expected_current_index
                or len(env.history.frames) != expected_current_index + 1):
            raise RuntimeError("reset-389 FrameHistory index/route-start contract is malformed")
        pinned_window = _pinned_original_389_pre_window(root)
        if not _same_window_exact(observation, pinned_window):
            raise RuntimeError("reset-389 policy input differs from pinned original_389 first PRE window")
        controls = None
        if env.config.get("control_edge_features", False):
            from prince_valiant_rl.control_history import ControlHistory
            controls = ControlHistory.from_recording(route.buttons, earliest, route.start,
                env.config.get("control_age_cap_frames", 120))
        encoded = {}
        for index, frame in enumerate(range(earliest, 390)):
            if index and controls is not None: controls.push(int(route.buttons[frame - 1]))
            previous = route.ram[frame - 1] if frame > route.start else None
            previous_mask = int(route.buttons[frame - 1]) if previous is not None else 0
            encoded[frame] = env.encode(route.ram[frame], previous, route.tracker(frame),
                previous_mask, previous is not None, controls=controls)
        from prince_valiant_rl.training.temporal import empty_observation
        blank = empty_observation(encoded[389])
        expected = {key: np.stack([encoded.get(389 + offset, blank)[key] for offset in offsets])
            for key in encoded[389]}
        expected["history_valid"] = np.asarray([389 + offset in encoded for offset in offsets], np.float32)
        if not _same_window_exact(expected, pinned_window):
            raise RuntimeError("reset-389 history/controller/feature input differs from trusted HumanRoute evidence")
        result = {"verified": True, "reset_frame": 389, "restoration_native_frames": 0,
            "ram_matches_route": True, "controller_matches_route": True,
            "history_slots": len(env.history.frames), "history_current_index": int(env.history.current),
            "invalid_pre_reset_slots": int((pinned_window["history_valid"] == 0).sum()),
            "pinned_reference_window_sha256": ORIGINAL_389_PRE_WINDOW_SHA256,
            "route_sha256": route.meta["route_sha256"],
            "policy_input_sha256": hashlib.sha256(b"".join(np.ascontiguousarray(observation[key]).tobytes()
                for key in sorted(observation))).hexdigest()}
        durable({"kind": "reset_provenance", **result})
        return result
    from pathlib import Path as _Path
    output_dir.mkdir(parents=True, exist_ok=True)
    if not label.replace("_", "").isalnum(): raise ValueError("unsafe milestone episode label")
    video_path = output_dir / f"milestone_{update:04d}_{label}.mp4"
    result = evaluate_reset389_video(policy, env_factory, identity_receipt=identity,
        reset_verifier=reset, output_path=video_path, receipt_fn=durable,
        checkpoint_update=update, checkpoint_sha256=checkpoint_sha256,
        frame_reader=screen_reader, encoder_factory=video_writer)
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_inputs(root: Path) -> dict[str, str]:
    cache = root / CACHE_REL
    actor, config = root / ACTOR_REL, root / CONFIG_REL
    paths = {"manifest": cache / "manifest.json", "READY": cache / "READY",
             "identity": cache / "identity.json", "actor": actor, "config": config}
    actual = {name: _sha256(path) for name, path in paths.items()}
    for name, expected in PINS.items():
        if actual[name] != expected:
            raise ValueError(f"pinned DQfD production input changed: {name}")
    ready = json.loads(paths["READY"].read_text(encoding="utf-8"))
    if ready.get("manifest_sha256") != actual["manifest"]:
        raise ValueError("v4 READY receipt is not linked to the pinned manifest")
    return actual


def create_fresh_demo_learner(*, root: Path, device: str = "cuda:0",
                              input_validator: Callable[..., Any] = _validate_inputs,
                              cache_factory: Callable[..., Any] = TransitionCache,
                              actor_loader: Callable[..., Any] = torch.load,
                              actor_factory: Callable[..., Any] = WindowPath,
                              model_factory: Callable[..., Any] = build_q_from_actor,
                              optimizer_factory: Callable[..., Any] = optimizer_for,
                              demo_index_factory: Callable[..., Any] = DemoReplayIndex,
                              learner_factory: Callable[..., Any] = DQfDBatchLearner) -> DQfDBatchLearner:
    """Create fresh encoder-transferred Q/target/Adam/RNG and protected demo PER.

    This factory deliberately has no online-store or environment parameter.
    The injected factories are for offline acceptance at these exact seams;
    they do not bypass this validation and assembly function.
    """
    root = Path(root).resolve()
    input_hashes = input_validator(root)
    cache = cache_factory(root / CACHE_REL)
    try:
        if int(cache.manifest.get("rows", -1)) != 21599:
            raise ValueError("pinned v4 accepted transition row count changed")
        if {row["name"]: int(row["rows"]) for row in cache.manifest["sources"]} != SOURCE_ROWS:
            raise ValueError("pinned v4 accepted source partition changed")
        if list(cache.manifest.get("history_offsets", ())) != [-60, -30, -15, -9, -6, -3, 0]:
            raise ValueError("pinned v4 history contract changed")
        actor_path = root / ACTOR_REL
        actor_payload = actor_loader(actor_path, map_location="cpu", weights_only=False)
        if actor_payload.get("architecture") != "separate_window_gru_branch_norm_v2" or int(actor_payload.get("actions", 0)) != 17:
            raise ValueError("pinned actor architecture/action contract changed")
        if not isinstance(actor_payload.get("shapes"), dict) or not isinstance(actor_payload.get("config"), dict):
            raise ValueError("pinned actor omitted model shapes/config")
        actor = actor_factory(actor_payload["shapes"], 17, actor_payload["config"])
        actor_state = {key.removeprefix("actor."): value for key, value in actor_payload["state_dict"].items()
                       if key.startswith("actor.")}
        actor.load_state_dict(actor_state, strict=True)
        torch.manual_seed(20261024)
        online, target = model_factory(actor, actor_payload["shapes"], 17,
                                       actor_payload["config"], 20261024, device)
        optimizer = optimizer_factory(online)
        demo_index = demo_index_factory(cache, gamma=.99997, n_step=10)
        learner = learner_factory(online, target, optimizer, demo_index, None,
            np.random.default_rng(20261024), successful_steps=0,
            phase="pretrain", phase_identity=input_hashes["manifest"], demo_only=True)
        learner.pinned_input_hashes = input_hashes
        learner.model_metadata = {"architecture": "dueling_window_dqn_v1",
            "shapes": {key: list(value) for key, value in actor_payload["shapes"].items()},
            "actions": 17, "model_config": actor_payload["config"],
            "actor_sha256": input_hashes["actor"], "config_sha256": input_hashes["config"],
            "manifest_sha256": input_hashes["manifest"], "READY_sha256": input_hashes["READY"],
            "identity_sha256": input_hashes["identity"], "seed": 20261024,
            "gamma": .99997, "batch_size": 128, "demo_only": True,
            "optimizer_contract": {"name": "Adam", "lr": 1e-4, "eps": 1e-8,
                "betas": [.9, .999], "weight_decay": 0.0}}
        return learner
    except BaseException:
        cache.close()
        raise
