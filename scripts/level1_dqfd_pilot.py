"""Native-free DQfD pilot preflight and explicit phase contract surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from _bootstrap import ROOT
from prince_valiant_rl.training.dqfd_transitions import TransitionCache, sha256_file
from prince_valiant_rl.training.dqfd_pretrain_entrypoint import dispatch_pretrain, private_child

CACHE = ROOT / "training/level1_dqfd_transitions_reconstruction_v4_20261024"
ACTOR = ROOT / "analysis/level1_full_route_verification_20261023/epoch400_eval_actor.pt"
CONFIG = ROOT / "training/level1_condition_B_score_quarantine_v1_20261024/ppo/config.json"
B_SNAPSHOT = ROOT / "training/level1_condition_B_anchored_recovery_v2_20261024/trusted_recovery_snapshot.pkl"
MANIFEST_PIN = "23c3b215462f2bf921a965a9e221e45e49447aafbb3b209da5dc30c9ee8d0a84"
READY_PIN = "bdc6e0c7f3688c56c255ab554479dc5de6a3ab39617cf7dfc4ca7857dcd00f8e"
IDENTITY_PIN = "a7590cafce66d89050638bdb44d63dd2767dad18622dbd0e376e6d9d0ce1683b"
SOURCE_ROWS = {"original_389": 18273, "human18000_noop2": 836,
               "human17940_noop2": 837, "ppo_noop1": 838, "ppo_noop16": 815}
PINS = {
    "actor": "780269fac004b733b3d8b002e8d4c9854f83958e83eeddce0d72b41e7fcab18e",
    "config": "f2a28db6efc046b8527fab61b39a21fff2de8e3762eedc4638f0eb8e193f6d7d",
    "B_snapshot": "c8bcb655dccb17c44eaf471aea216bad44ba9fef26d58113f3c01308e1af485b",
}


def main(argv=None, *, dependencies=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "pretrain"))
    parser.add_argument("--output", type=Path, help="fresh implementation report root")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--protocol-sha256", default=None)
    parser.add_argument("--execution-budget", type=Path,
        help="explicit hash-linked current execution-budget receipt required for pretrain")
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="validate the full child launch without spawning it")
    parser.add_argument("--_private-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--envelope", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    parser.add_argument("--_attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--_execution-budget", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_execution-budget-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--help-contract", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._private_child:
        if args.phase != "pretrain" or args.envelope is None or not args.token:
            raise SystemExit("invalid private pretraining child invocation")
        return private_child(envelope_path=args.envelope, supplied_token=args.token,
                             expected_attempt_id=args._attempt_id,
                             expected_execution_budget_path=args._execution_budget,
                             expected_execution_budget_sha256=args._execution_budget_sha256,
                             dependencies=dependencies)
    if args.help_contract:
        print(json.dumps({"status": "contract_only_live_phases_currently_blocked",
            "pretrain": {"updates": 2000, "batch_size": 128, "milestone_updates": [1000, 2000],
                "demo_only": True, "target_copies": [1000, 2000]},
            "pretraining_demos": {"condition": "greedy reset389", "episodes_max": 3,
                "native_frames_max": 108000, "episode_horizon_max": 36000,
                "active_seconds_max": 900, "extra_old_best_demo_only_if_update1000_remains_best": True},
            "pretraining_model_io_seconds_max": 600,
            "video_bundle_bytes_max_each": 100 * 1024 * 1024,
            "video_bundle_bytes_max_total": 300 * 1024 * 1024,
            "online": "blocked; conditional <=368640 training frames, expected 5696 updates",
            "future_online_demo_updates": [3000, 4000, 5000, 6000, 7000],
            "final-eval": "blocked; fixed B, original-route, corrected-C endpoints remain separate",
            "planned_pretrain_command_not_executable": ".venv-gpu\\Scripts\\python.exe run.py level1_dqfd_pilot pretrain --output training/level1_dqfd_pilot_seed20261024",
            "planned_retention_command_not_executable": ".venv-gpu\\Scripts\\python.exe run.py level1_dqfd_pilot retention --output training/level1_dqfd_pilot_seed20261024"}, indent=2))
        return 0
    if args.phase == "pretrain":
        if args.output is None:
            raise SystemExit("pretrain requires an explicit fresh --output")
        if args.execution_budget is None:
            raise SystemExit("pretrain requires an explicit --execution-budget receipt; no launch-ready default exists")
        protocol_path = ROOT / "analysis/dqfd_pilot_worker_prompts_20261024/11_WIRE_PRETRAIN_PRODUCTION_CLI_PROMPT.txt"
        actual_protocol_hash = "74a7f1744688c7c7c6b6f57c059a266f8329d0d14a08c2544163552e9a4880ed"
        protocol_hash = args.protocol_sha256 or actual_protocol_hash
        if protocol_hash != actual_protocol_hash:
            raise SystemExit("protocol hash differs from the preserved prompt11 input")
        report_dir = args.report_dir or ROOT / "analysis/level1_dqfd_pretrain_entrypoint_20261024"
        result = dispatch_pretrain(root=ROOT, output=args.output, config=args.config, cache=args.cache,
            protocol_sha256=protocol_hash, execution_budget_path=args.execution_budget,
            dry_run=args.dry_run,
            report_dir=report_dir, dependencies=dependencies)
        print(json.dumps(result, indent=2))
        return 0
    if args.phase != "preflight":
        raise SystemExit(f"{args.phase} execution is not implemented/authorized by the tested CLI; no environment or worker was created")
    if args.output is not None and args.output.exists():
        raise SystemExit(f"fresh output is occupied: {args.output}")
    hashes = {"actor": sha256_file(ACTOR), "config": sha256_file(CONFIG), "B_snapshot": sha256_file(B_SNAPSHOT),
              "manifest": sha256_file(CACHE / "manifest.json"), "READY": sha256_file(CACHE / "READY"),
              "identity": sha256_file(CACHE / "identity.json")}
    for name in ("actor", "config", "B_snapshot"):
        if hashes[name] != PINS[name]: raise SystemExit(f"pinned {name} hash mismatch")
    for name, expected in (("manifest", MANIFEST_PIN), ("READY", READY_PIN), ("identity", IDENTITY_PIN)):
        if hashes[name] != expected: raise SystemExit(f"transition {name} pin mismatch")
    ready = json.loads((CACHE / "READY").read_text(encoding="utf-8"))
    if ready.get("manifest_sha256") != MANIFEST_PIN: raise SystemExit("READY linkage mismatch")
    with TransitionCache(CACHE) as cache:
        counts = {s["name"]: int(s["rows"]) for s in cache.manifest["sources"]}
        if counts != SOURCE_ROWS: raise SystemExit("transition source names/counts mismatch")
        if tuple(cache.manifest.get("history_offsets", ())) != (-60, -30, -15, -9, -6, -3, 0):
            raise SystemExit("history offset mismatch")
    result = {"status": "preflight_passed_native_free", "hashes": hashes, "source_rows": counts,
              "manifest_rows": 21599, "history_offsets": [-60, -30, -15, -9, -6, -3, 0],
              "native_owner_imported": False, "environment_workers": 0, "game_frames": 0,
              "optimizer_steps": 0, "gameplay_evaluations": 0,
              "phase_execution_ready": False, "pretraining_execution_ready": False,
              "milestone1000_2000_demo_ready": False, "retention_ready": False,
              "online_collector_ready": False, "final_evaluator_ready": False,
              "whole_trainer_ready": False,
              "component_readiness": {"protected_demo_gather": True,
                  "online_store_under_pinned_trusted_B_contract": True,
                  "demo_only_learner_actual_update_path": True,
                  "pretraining_milestone_loop": True,
                  "reset389_evaluator_and_video_writer": True,
                  "gpu_resource_sampler_and_owned_child_supervisor": True,
                  "production_pretraining_factory_wired_to_cli": True,
                  "production_environment_identity_and_reset_verifier": True,
                  "production_demo_executor_wired_to_milestone_loop": True,
                  "production_collector": False, "final_evaluator": False,
                  "phase_execution_ready": False}}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
