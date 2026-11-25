"""Bounded child process for trusted diagnostic workloads."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter, process_time

from quinn.adapters import kernelbench
from quinn.measurement import cuda_event_ms, measure_block, sequence
from quinn.telemetry import sample
from quinn.workloads import build_variant


def utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def background(options, write):
    mode = options.get("interference", "none")
    if mode == "none":
        yield
        return
    duty = str(options.get("contention_duty", 0.5))
    process = subprocess.Popen([sys.executable, "-m", "quinn.workers.contention", mode, duty],
                               stdout=subprocess.PIPE, text=True)
    try:
        if process.stdout.readline().strip() != "READY":
            raise RuntimeError("Contention process failed during warm-up")
        write({"record": "interference", "mode": mode, "requested_host_duty": float(duty),
               "period_ms": 40, "pid": process.pid})
        yield
        if process.poll() is not None:
            raise RuntimeError("Contention process exited during capture")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()


def run_timing(spec, write, torch, device, args, reference, candidate):
    row, options = spec["row"], spec["options"]
    rng = random.Random(f'{spec["seed"]}:{row["candidate"]}')
    mode, count = options["mode"], options["blocks"]
    launches = options["launches_per_event"]
    if mode == "b0":
        upstream_timer = kernelbench.load(
            Path(options["upstream"]), "src/kernelbench/timing.py", "quinn_pinned_timing"
        ).time_execution_with_cuda_event
        start = perf_counter()
        candidate_times = upstream_timer(candidate, args, num_trials=count, verbose=False, device=device)
        ref_times = upstream_timer(reference, args, num_trials=count, verbose=False, device=device)
        write({"record": "upstream_b0", "reference_ms": ref_times, "candidate_ms": candidate_times,
               "timing_wall_ms": (perf_counter() - start) * 1000, "utc": utc()})
        return

    programs = {"R": reference, "C": candidate}

    def timer(role):
        return cuda_event_ms(torch, device, lambda: programs[role](*args), launches=launches)

    def telemetry(index, phase):
        if options["telemetry_every"] and index % options["telemetry_every"] == 0:
            write({"record": "telemetry", "phase": phase, "before_block": index,
                   **sample(spec["gpu_uuid"])})

    def warm(roles):
        warmup = 3 if mode == "m0" else options["warmup"]
        start = perf_counter()
        for _ in range(warmup):
            for role in roles:
                programs[role](*args)
        torch.cuda.synchronize(device)
        write({"record": "warmup", "roles": roles, "iterations_per_program": warmup,
               "wall_ms": (perf_counter() - start) * 1000})

    if mode in {"m0", "m1"}:
        for role in ("R", "C"):
            warm(role)
            for index in range(count):
                telemetry(index, role)
                start = perf_counter()
                times = [timer(role), timer(role)]
                if any(value <= 0 for value in times):
                    raise ValueError("nonpositive CUDA event timing")
                write({"record": "fixed_block", "block": index, "role": role, "utc": utc(),
                       "samples_ms": times, "launches_per_event": launches,
                       "block_wall_ms": (perf_counter() - start) * 1000})
        return

    warm("RC")
    for index in range(count):
        telemetry(index, "paired")
        block = measure_block(sequence(mode, rng, options["order"]), timer, launches)
        write({"record": "block", "block": index, "utc": utc(), **block})


def capture(spec, write):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one visible CUDA device")
    device = torch.device("cuda:0")
    if "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError("CUDA worker requires an A100")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    row, options = spec["row"], spec["options"]
    root = Path(options["upstream"])
    workload = kernelbench.load(root, row["path"], "quinn_pinned_task")
    rng = random.Random(spec["seed"])
    torch.manual_seed(spec["fixed_input_seed"])
    init = workload.get_init_inputs()
    reference = workload.Model(*init).to(device).eval()
    candidate_model = workload.Model(*init).to(device).eval()
    candidate_model.load_state_dict(reference.state_dict())

    def inputs(seed):
        torch.manual_seed(seed)
        return [x.to(device) if isinstance(x, torch.Tensor) else x for x in workload.get_inputs()]

    args = inputs(spec["fixed_input_seed"])
    details = [{"shape": list(x.shape), "dtype": str(x.dtype)}
               if isinstance(x, torch.Tensor) else {"value": x} for x in args]
    with torch.inference_mode():
        start, cpu_start = perf_counter(), process_time()
        try:
            candidate = build_variant(torch, candidate_model, args, row)
            candidate(*args)
            torch.cuda.synchronize(device)
        except Exception as error:
            write({"record": "validation", "compilation_status": "failed",
                   "correctness_status": "unverified", "error": repr(error),
                   "setup_wall_ms": (perf_counter() - start) * 1000,
                   "setup_self_cpu_ms": (process_time() - cpu_start) * 1000,
                   "inputs": details})
            return
        write({"record": "compilation", "status": "passed",
               "setup_wall_ms": (perf_counter() - start) * 1000,
               "setup_self_cpu_ms": (process_time() - cpu_start) * 1000})
        validation_start = perf_counter()
        for _ in range(options["correctness_trials"]):
            trial_seed = rng.randrange(2**32)
            trial = inputs(trial_seed)
            expected = reference(*trial)
            independent = [x.clone() if isinstance(x, torch.Tensor) else x for x in trial]
            observed = candidate(*independent)
            torch.cuda.synchronize(device)
            mutates = any(not torch.equal(x, y) for x, y in zip(trial, independent)
                          if isinstance(x, torch.Tensor))
            tolerance = kernelbench.FP32_TOLERANCE
            passes = (isinstance(observed, torch.Tensor) and expected.shape == observed.shape
                      and expected.dtype == observed.dtype and not mutates
                      and bool(torch.allclose(expected, observed, atol=tolerance, rtol=tolerance)))
            write({"record": "correctness_trial", "seed": trial_seed, "passes": passes,
                   "input_mutation": mutates})
            if not passes:
                write({"record": "validation", "compilation_status": "passed",
                       "correctness_status": "failed", "inputs": details,
                       "correctness_wall_ms": (perf_counter() - validation_start) * 1000})
                return
        write({"record": "validation", "compilation_status": "passed",
               "correctness_status": "passed", "inputs": details,
               "correctness_wall_ms": (perf_counter() - validation_start) * 1000})
        with background(options, write):
            run_timing(spec, write, torch, device, args, reference, candidate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("output")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    with Path(args.output).open("x", buffering=1) as output:
        valid = False

        def write(record):
            nonlocal valid
            if record["record"] == "validation":
                valid = record["correctness_status"] == "passed"
            output.write(json.dumps(record, allow_nan=False) + "\n")

        write({"record": "start", "utc": utc(), "pid": os.getpid(), "spec": spec})
        capture(spec, write)
        if spec["options"].get("require_correct") and not valid:
            raise RuntimeError("Smoke workload did not pass compilation and correctness")
        write({"record": "complete", "utc": utc()})


if __name__ == "__main__":
    main()
