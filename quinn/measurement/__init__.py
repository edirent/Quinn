"""CUDA event timing and order-aware reference/candidate blocks."""
from math import log
from statistics import mean
from time import perf_counter


def sequence(mode, rng, block):
    if mode in {"m2", "gold"}:
        return block if block != "random" else rng.choice(["RCCR", "CRRC"])
    raise ValueError("unknown capture mode")


def cuda_event_ms(torch, device, call, launches=1):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        output = call()
    end.record()
    torch.cuda.synchronize(device)
    elapsed = start.elapsed_time(end) / launches
    del output
    return elapsed


def measure_block(order, timer, launches_per_event=1):
    start = perf_counter()
    samples = [{"role": role, "ms": timer(role)} for role in order]
    elapsed = (perf_counter() - start) * 1000
    runtimes = {role: [point["ms"] for point in samples if point["role"] == role]
                for role in ("R", "C")}
    if any(value <= 0 for values in runtimes.values() for value in values):
        raise ValueError("nonpositive event timing")
    return {"order": order, "samples": samples, "log_speedup": log(mean(runtimes["R"]))
            - log(mean(runtimes["C"])), "event_sum_ms": launches_per_event * sum(point["ms"] for point in samples),
            "launches_per_event": launches_per_event,
            "block_wall_ms": elapsed}
