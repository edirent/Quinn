"""Coordinator for independent bounded captures on one or two devices."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import median
import re
import random
import signal
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
from time import perf_counter

from quinn.telemetry import a100_devices, hardware_report
from quinn.workers.budget import Ledger
from quinn.workloads import manifest


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def write_json(path, value, replace=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x") as output:
            output.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        return
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        staging = Path(output.name)
        try:
            output.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
            output.flush()
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)


def capture(options, indices, budget_hours, output, resume):
    hardware = hardware_report()
    hardware_path = output / "preflight.json"
    if not (output / "run.json").exists():
        write_json(hardware_path, hardware)
    devices = a100_devices(indices)
    if not hardware["packages"]["torch"]:
        raise ValueError("CUDA Python dependencies missing. Run make setup-gpu.")
    if not shutil.which("timeout"):
        raise ValueError("GPU worker requires coreutils timeout for an independent watchdog.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", options["session"]):
        raise ValueError("Invalid session identifier")
    batch_duration = options.get("batch_deadline_s")
    if batch_duration is not None and (not math.isfinite(batch_duration) or batch_duration <= 2):
        raise ValueError("Batch deadline must exceed the cleanup reserve")
    mode, order = options["mode"], options["order"]
    if mode not in {"b0", "m0", "m1", "m2", "gold"} or order not in {"random", "RCCR", "CRRC", "RCRC"}:
        raise ValueError("Invalid measurement protocol or pairing order")
    if "condition" not in options or not str(options["condition"]).strip():
        raise ValueError("Supply an explicit capture condition label")
    if min(options[k] for k in ("blocks", "warmup", "correctness_trials", "launches_per_event")) < 1:
        raise ValueError("Capture counts must be positive")
    if options["timeout_s"] <= 2 or options["telemetry_every"] < 0:
        raise ValueError("Capture timeout must exceed cleanup reserve; telemetry period must be nonnegative")
    interference = options.get("interference", "none")
    if interference not in {"none", "compute", "memory"}:
        raise ValueError("Unknown interference condition")
    if options.get("contention_duty", 0.5) not in {0.25, 0.5, 0.75}:
        raise ValueError("Contention duty must be 0.25, 0.5 or 0.75")
    if mode == "gold" and interference != "none":
        raise ValueError("Gold capture requires the isolated target")
    if options["blocks"] < 2:
        raise ValueError("Use at least two blocks")
    rows = manifest(options["upstream"])
    tasks = {}
    for row in rows:
        tasks.setdefault(row["task"], []).append(row)
    selected = list(tasks)[:options.get("task_limit") or len(tasks)]
    if options.get("tasks"):
        selected = options["tasks"]
        if len(set(selected)) != len(selected) or any(key not in tasks for key in selected):
            raise ValueError("Unknown or repeated task ID")
    corpus = [row for task in selected for row in tasks[task]
              if not options.get("variants") or row["variant"] in options["variants"]]
    if not corpus:
        raise ValueError("Empty capture selection")
    source_root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(source_root)): sha256(p.read_bytes()).hexdigest()
               for p in sorted(source_root.rglob("*.py"))}
    immutable = {"type": "a100_diagnostic_capture", "options": options, "devices": devices,
                 "platform": hardware["platform"], "python": hardware["python"],
                 "packages": hardware["packages"], "sources": sources, "corpus": corpus,
                 "budget_hours": budget_hours}
    ceiling = budget_hours * 3600
    ledger = Ledger(options["ledger"], ceiling)
    workers = [[] for _ in devices]
    task_slots = {task: index % len(devices) for index, task in enumerate(tasks)}
    randomized = corpus.copy()
    random.Random(options["seed"]).shuffle(randomized)
    for row in randomized:
        workers[task_slots[row["task"]]].append(row)

    with ExitStack() as stack:
        for uuid in sorted(device["uuid"] for device in devices):
            lease = stack.enter_context(Path(f"/tmp/quinn-{uuid}.lease").open("a"))
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"GPU lease held by another Quinn capture: {uuid}") from exc
        run_path = output / "run.json"
        if run_path.exists():
            if not resume or json.loads(run_path.read_text()) != immutable:
                raise ValueError("Existing run differs or --resume omitted. Use a new output directory.")
        else:
            write_json(run_path, immutable, replace=False)
        epoch = datetime.now(timezone.utc).isoformat()
        write_json(output / "preflight.json", hardware)
        started = perf_counter()
        deadline = started + (batch_duration or math.inf)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            leases = {uuid: stack.enter_context(Path(f"/tmp/quinn-{uuid}.child").open("a"))
                      for uuid in (d["uuid"] for d in devices)}
            futures = [pool.submit(run_device, gpu, rows, options, ledger, output, resume,
                                   leases[gpu["uuid"]], deadline, started)
                       for gpu, rows in zip(devices, workers)]
            results = []
            errors = []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append(str(exc))
            completed = []
            for gpu, rows in zip(devices, workers):
                for row in rows:
                    key = job_key(output, row, gpu["uuid"], options)
                    attempt = ledger.completed_attempt(key)
                    if attempt is not None:
                        path = job_directory(output, row) / f"{attempt}.jsonl"
                        if closed_capture(path):
                            completed.append(str(path.relative_to(output)))
            wall = perf_counter() - started
            timings = [item for result in results for item in result["candidate_completions"]]
            expected = Counter(item["task"] for item in corpus)
            finished = Counter(item["task"] for item in timings)
            tasks_done = [task for task in finished if finished[task] == expected[task]]
            latencies = sorted(item["latency_s"] for item in timings)
            report = {**ledger.summary(), "completed_artifacts": completed,
                      "wall_seconds_this_invocation": wall, "invocation_utc": epoch,
                      "worker_summary": results, "errors": errors,
                      "candidate_requests_per_second": len(latencies) / wall,
                      "complete_groups_per_second": len(tasks_done) / wall,
                      "p50_candidate_latency_s": median(latencies) if latencies else None,
                      "p95_candidate_latency_s": latencies[math.ceil(len(latencies) * .95) - 1]
                      if latencies else None}
            write_json(output / "accounting.json", report)
            if errors or any(result["failures"] or result["deadline_reached"] for result in results):
                raise ValueError(f"Capture incomplete. See {output / 'accounting.json'}")
    return report


def run_device(device, rows, options, ledger, output, resume, lease, deadline, batch_start):
    uuid = device["uuid"]
    # The child inherits the second lock, which survives death of the coordinator.
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    completed = failures = skipped = 0
    busy_s = 0.0
    completion_times = []
    deadline_reached = False
    started = perf_counter()
    for row in rows:
        candidate = row["candidate"]
        job = job_key(output, row, uuid, options)
        base = job_directory(output, row)
        raw_completed = {int(raw.stem) for raw in base.glob("*.jsonl") if closed_capture(raw)}
        ledger.recover(job, raw_completed)
        remaining = deadline - perf_counter()
        if remaining <= 3:
            deadline_reached = True
            break
        reservation_s = min(options["timeout_s"], remaining)
        attempt = ledger.reserve(job, "reward_evaluation", reservation_s, resume)
        if attempt is None:
            previous = ledger.completed_attempt(job)
            if previous not in raw_completed:
                raise ValueError(f"Ledger references missing or incomplete raw capture for {candidate}")
            skipped += 1
            continue
        base.mkdir(parents=True, exist_ok=True)
        seed = int(sha256(encoded([options["seed"], row["task"]]).encode()).hexdigest()[:8], 16)
        spec = base / f"{attempt}.spec.json"
        raw = base / f"{attempt}.jsonl"
        fixed_input_seed = int(sha256(encoded([options.get("input_seed", 141421), row["task"]])
                                     .encode()).hexdigest()[:8], 16)
        write_json(spec, {"row": row, "options": options, "gpu_uuid": uuid, "seed": seed,
                          "fixed_input_seed": fixed_input_seed}, replace=False)
        # Only pinned reference and maintained candidates are accepted by this worker.
        keep = ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT")
        env = {key: os.environ[key] for key in keep if key in os.environ}
        cache = (output / "worker-cache" / uuid).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        env.update(CUDA_VISIBLE_DEVICES=uuid, HOME=str(cache), XDG_CACHE_HOME=str(cache),
                   OMP_NUM_THREADS="1", TORCHINDUCTOR_COMPILE_THREADS="1")
        started_job = perf_counter()
        success = False
        detail = "child launch was interrupted"
        process = None
        try:
            with (base / f"{attempt}.capture.log").open("x") as log:
                duration = reservation_s - 2
                process = subprocess.Popen(["timeout", "--kill-after=1s", f"{duration}s",
                                            sys.executable, "-m", "quinn.workers.cuda_capture",
                                            str(spec), str(raw)], env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(lease.fileno(),))
                try:
                    process.wait(timeout=max(0.01, reservation_s - 0.5 - (perf_counter() - started_job)))
                    success = process.returncode == 0 and closed_capture(raw)
                    detail = f"child exit {process.returncode}"
                except subprocess.TimeoutExpired:
                    detail = "worker timeout"
        finally:
            try:
                if process is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            finally:
                elapsed = perf_counter() - started_job
                ledger.finish(attempt, elapsed, success, detail)
                busy_s += elapsed
        if success:
            completed += 1
            completion_times.append({"task": row["task"], "candidate": candidate,
                                     "latency_s": perf_counter() - batch_start})
        else:
            failures += 1
        print(encoded({"candidate": candidate, "gpu_uuid": uuid, "attempt": attempt,
                       "status": detail, "charged_seconds": elapsed}), flush=True)
    return {"gpu_uuid": uuid, "completed": completed, "skipped": skipped, "failures": failures,
            "worker_assigned_seconds": busy_s, "candidate_completions": completion_times,
            "deadline_reached": deadline_reached,
            "worker_idle_seconds": max(0.0, perf_counter() - started - busy_s)}


def job_key(output, row, uuid, options):
    return sha256(encoded([str(output.resolve()), row["candidate"], uuid, options]).encode()).hexdigest()


def job_directory(output, row):
    return output / "jobs" / row["task"] / row["variant"]


def closed_capture(path):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 512))
        lines = f.read().splitlines()
    if not lines:
        return False
    try:
        return json.loads(lines[-1]).get("record") == "complete"
    except (ValueError, AttributeError):
        return False
