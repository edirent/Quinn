"""Configured command-line entry points."""
import argparse
from pathlib import Path

import yaml

from quinn import diagnostics
from quinn.telemetry import hardware_report
from quinn.workers import capture, encoded, write_json
from quinn.workloads import manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m quinn.run")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--gpu-hour-budget", type=float, default=96)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="Reanalyze saved synthetic traces")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--session")
    parser.add_argument("--mode", choices=["b0", "m0", "m1", "m2", "gold"])
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--condition")
    parser.add_argument("--batch-deadline-s", type=float)
    parser.add_argument("--replay-budgets-ms", nargs="+", type=float)
    parser.add_argument("--block-bound-ms", type=float)
    parser.add_argument("--telemetry-every", type=int)
    parser.add_argument("--interference", choices=["none", "compute", "memory"])
    parser.add_argument("--contention-duty", type=float, choices=[0.25, 0.5, 0.75])
    parser.add_argument("--hardware-check", action="store_true")
    parser.add_argument("--corpus", type=Path, help="Path to the pinned KernelBench checkout")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--analyze-capture", nargs="+", type=Path, metavar="RUN")
    args = parser.parse_args(argv)
    try:
        if args.hardware_check:
            info = hardware_report()
            path = args.output_dir or Path("artifacts/manifests")
            write_json(path / "hardware.json", info)
            print(encoded({"path": str(path / "hardware.json"), "gpus": info["gpus"],
                           "clock_lock_permissions": info["clock_lock_permissions"]}))
            return 0
        if args.corpus:
            rows = manifest(args.corpus)
            path = args.output_dir or Path("artifacts/manifests")
            write_json(path / "pilot_corpus.json", rows)
            print(encoded({"tasks": len({row["task"] for row in rows}), "recipes": len(rows),
                           "path": str(path / "pilot_corpus.json")}))
            return 0
        if args.report:
            from quinn.reporting import publication
            result = publication(args.output_dir or Path("artifacts/diagnostics"))
            print(encoded(result))
            return 0
        if args.analyze_capture:
            from quinn.reporting import capture_analysis
            if args.replay_budgets_ms and args.block_bound_ms is None:
                parser.error("--replay-budgets-ms requires a declared --block-bound-ms")
            result = capture_analysis(args.analyze_capture,
                                      args.output_dir or Path("artifacts/processed/a100"),
                                      args.replay_budgets_ms, args.block_bound_ms)
            print(encoded(result))
            return 0
        if args.config is None:
            parser.error("--config, --hardware-check, --corpus, --analyze-capture or --report is required")
        config = yaml.safe_load(args.config.read_text())
        for name in ("seed", "session", "mode", "task_limit", "blocks", "condition", "telemetry_every",
                     "interference", "contention_duty", "batch_deadline_s"):
            if getattr(args, name) is not None:
                config[name] = getattr(args, name)
        kind = config.pop("kind")
        if kind == "capture":
            if config.get("task_limit") is not None and config["task_limit"] < 1:
                raise ValueError("task limit must be positive")
            output = args.output_dir or Path("artifacts/raw") / config["session"]
            report = capture(config, args.gpus.split(","), args.gpu_hour_budget, output, args.resume)
            print(encoded(report))
        elif kind == "synthetic":
            output = args.output_dir or Path("artifacts/diagnostics")
            result = diagnostics.run(config, output, resume=args.resume, recompute=args.recompute)
            print(encoded({"evidence": result["evidence"], "runs": len(result["runs"])}))
        elif kind == "online_rl":
            raise ValueError("Online RL is gated. A100 evidence for Gates A–C, a restricted "
                             "execution environment and a verified trainer integration are pending.")
        else:
            raise ValueError(f"Unknown experiment kind {kind}")
    except (ValueError, RuntimeError, OSError, KeyError, yaml.YAMLError) as exc:
        print(encoded({"status": "blocked" if args.config is not None else "error", "reason": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
