"""Predeclared KernelBench diagnostic panel and trusted candidate recipes."""
from hashlib import sha256
from pathlib import Path

from quinn.adapters import kernelbench

PILOT = {
    "matrix multiplication": {1: [1, 2, 3, 4, 8, 9]},
    "elementwise": {1: [19, 20, 21, 22, 25, 26]},
    "softmax": {1: [23, 24]},
    "normalization": {1: [33, 35, 36, 40]},
    "reduction": {1: [47, 48, 49]},
    "convolution": {1: [50, 67, 87]},
    "fused": {2: [1, 9, 12, 22, 29, 40]},
}
HELD_OUT = {(1, 8), (1, 22), (1, 24), (1, 40), (1, 49), (1, 87), (2, 40)}
VARIANTS = ("eager-output-clone", "jit-trace", "jit-frozen", "inductor-default",
            "inductor-reduce-overhead", "inductor-max-autotune-no-cudagraphs")
TRITON_VARIANTS = ("triton-128", "triton-256", "triton-512")
TRITON_TASKS = {19, 21, 22}


def manifest(root):
    root = Path(root)
    sources = {}
    for path in kernelbench.files(root):
        name = Path(path)
        if name.parent.name not in {"level1", "level2"} or name.suffix != ".py":
            continue
        sources[(int(name.parent.name[-1]), int(name.name.split("_")[0]))] = path
    recipe = Path(__file__).read_bytes()
    triton_recipe = Path(__file__).with_name("triton_ops.py").read_bytes()
    rows = []
    for category, tasks in PILOT.items():
        for level, ids in tasks.items():
            for task_id in ids:
                path = sources[level, task_id]
                upstream_code = kernelbench.source(root, path)
                task = f"L{level}-{task_id:03d}"
                triton = level == 1 and task_id in TRITON_TASKS
                variants = (*TRITON_VARIANTS, "jit-frozen", *VARIANTS[3:5]) if triton else VARIANTS
                for variant in variants:
                    code = triton_recipe if variant.startswith("triton-") else recipe
                    rows.append({"task": task, "candidate": f"{task}/{variant}", "variant": variant,
                                 "category": category, "path": path,
                                 "split": "held_out" if (level, task_id) in HELD_OUT else "development",
                                 "dataset_revision": kernelbench.REVISION,
                                 "reference_sha256": sha256(upstream_code).hexdigest(),
                                 "candidate_recipe_sha256": sha256(upstream_code + code + variant.encode()).hexdigest(),
                                 "source": "quinn_controlled_recipe", "rollout_group": None,
                                 "checkpoint": None, "generation_settings": None,
                                 "language": "triton" if variant.startswith("triton-") else "pytorch",
                                 "compile_status": "pending", "correctness_status": "pending"})
    return rows


def build_variant(torch, model, args, row):
    variant = row["variant"]
    if variant.startswith("triton-"):
        from quinn.workloads.triton_ops import candidate
        task = int(row["task"].split("-")[1])
        return candidate(task, int(variant.split("-")[1]))
    if variant == "eager-output-clone":
        return lambda *inputs: model(*inputs).clone()
    if variant.startswith("jit"):
        traced = torch.jit.trace(model, args, check_trace=False)
        return torch.jit.freeze(traced) if variant == "jit-frozen" else traced
    return torch.compile(model, mode=variant.removeprefix("inductor-"))
