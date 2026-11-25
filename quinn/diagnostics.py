"""Explicitly synthetic scanner for scheduler behavior and falsification checks."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from quinn.advantages import Advantage, propagated_variance
from quinn.estimation import Observation, Reward
from quinn.scheduling import Candidate, Policy, schedule
from quinn.scheduling.replay import Replay
from quinn.workers import encoded, write_json

EVIDENCE = "synthetic_cpu_diagnostic"
POLICIES = {
    "B3 Fixed paired": ("paired", Policy("fixed")),
    "B4 Reward variance": ("paired", Policy("reward")),
    "B5 Rank": ("paired", Policy("rank")),
    "B6 Quinn": ("paired", Policy("advantage")),
    "Quinn-NoPairing": ("unpaired", Policy("advantage")),
    "Quinn-NoCostAwareness": ("paired", Policy("advantage", cost_aware=False)),
    "Quinn-NoCovariance": ("paired", Policy("advantage", covariance=False)),
    "Quinn-NoDeadline": ("paired", Policy("advantage")),
}


def ar1(rng, count, rho):
    points = rng.standard_normal(count)
    for i in range(1, count):
        points[i] = rho * points[i - 1] + np.sqrt(1 - rho**2) * points[i]
    return points


def generate(seed, groups_count, blocks):
    rng = np.random.default_rng(seed)
    groups = {}
    gold = {}
    for i in range(groups_count):
        key = f"group-{i:03d}"
        n = (4, 8, 16)[i % 3]
        case = ("separated", "near_tie", "mixed_correctness", "heavy_tail")[(i // 3) % 4]
        spread = 0.012 if case == "near_tie" else 0.18
        correct = np.ones(n, dtype=bool)
        if case == "mixed_correctness":
            correct[rng.choice(n, size=max(1, n // 4), replace=False)] = False
        truth = rng.normal(0.03, spread, n)
        noise = np.exp(rng.uniform(np.log(0.018), np.log(0.14), n))
        cost = np.exp(rng.uniform(np.log(0.45), np.log(1.8), n))
        drift = ar1(rng, blocks, 0.82)
        groups[key] = {"case": case, "candidates": [], "bounds": {},
                       "paired": {}, "unpaired": {}}
        gold[key] = list(truth)
        for j in range(n):
            candidate = f"{key}/{j:02d}"
            item = groups[key]
            item["candidates"].append(asdict(Candidate(candidate, key, bool(correct[j]))))
            item["bounds"][candidate] = [float(cost[j]), float(cost[j] + 0.09)]
            local = noise[j] * ar1(rng, blocks, 0.35)
            coupling = rng.uniform(0.6, 1.4)
            if case == "heavy_tail":
                local += rng.binomial(1, 0.05, blocks) * rng.normal(0, 5 * noise[j], blocks)
            for label, factor in (("paired", 0.018), ("unpaired", 0.085)):
                series = truth[j] + local + factor * coupling * drift
                item[label][candidate] = [
                    asdict(Observation(k, float(observed), *item["bounds"][candidate]))
                    for k, observed in enumerate(series) if correct[j]
                ]
    return {"evidence": EVIDENCE, "seed": seed, "groups": groups}, gold


def fidelity(estimated, reference, advantage, unresolved=0.05):
    estimated, reference = np.asarray(estimated), np.asarray(reference)
    a, b = advantage.values(estimated), advantage.values(reference)
    resolved = np.abs(b) > unresolved
    i, j = np.triu_indices(len(a), 1)
    ranks = np.sign(reference[i] - reference[j])
    distinct = ranks != 0
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    result = {"advantage_rmse": float(np.sqrt(np.mean((a - b)**2))),
              "reward_rmse": float(np.sqrt(np.mean((estimated - reference)**2))),
              "sign_error": float(np.mean(np.sign(a[resolved]) != np.sign(b[resolved]))) if resolved.any() else None,
              "resolved_signs": int(resolved.sum()),
              "rank_error": float(np.mean(np.sign(estimated[i[distinct]] - estimated[j[distinct]])
                                          != ranks[distinct])) if distinct.any() else None,
              "cosine": float(np.dot(a, b) / norm) if norm else None,
              "spearman": None, "kendall": None}
    if np.ptp(estimated) and np.ptp(reference):
        result["spearman"] = float(spearmanr(estimated, reference).statistic)
        result["kendall"] = float(kendalltau(estimated, reference).statistic)
    return result


def simulate(config, trace, gold):
    reward = Reward(config["reward"])
    advantage = Advantage()
    for group, item in trace["groups"].items():
        candidates = [Candidate(**entry) for entry in item["candidates"]]
        correct = np.array([candidate.correct for candidate in candidates])
        cost = sum(item["bounds"][c.key][0] for c in candidates if c.correct)
        reference_logits = gold[group]
        for multiplier in config["budgets"]:
            for name, (channel, template) in POLICIES.items():
                policy = replace(template, initial=config["initial"])
                deadline = cost * multiplier * config["deadline_scale"]
                if name == "Quinn-NoDeadline":
                    deadline = float("inf")
                source = Replay(item[channel], item["bounds"])
                result = schedule(candidates, source, cost * multiplier, deadline,
                                  policy, reward, advantage)
                estimates = result.pop("estimates")
                counts = Counter(result.pop("events"))
                row = {"group": group, "case": item["case"], "group_size": len(candidates),
                       "method": name, "budget_level": multiplier,
                       "gpu_ms": result["gpu_ms"], "gpu_cap_ms": cost * multiplier,
                       "wall_ms": result["wall_ms"], "deadline_ms": deadline if np.isfinite(deadline) else None,
                       "scheduler_cpu_ms": result["scheduler_cpu_ms"],
                       "blocks": sum(counts.values()),
                       "counts": [counts[c.key] for c in candidates if c.correct],
                       "complete": not result["incomplete_groups"]}
                if row["complete"]:
                    reference = reward.values(reference_logits, correct)
                    observed = np.array([estimates[c.key].reward_mean for c in candidates])
                    row.update(fidelity(observed, reference, advantage))
                    row["coverage"] = float(np.mean([
                        estimates[c.key].lower_bound <= r <= estimates[c.key].upper_bound
                        for c, r in zip(candidates, reference) if c.correct
                    ]))
                yield row


def approximation(seed):
    rng = np.random.default_rng(seed)
    correlation = 0.65
    cov = (np.eye(4) * (1 - correlation) + correlation) * 0.01**2
    noise = rng.multivariate_normal(np.zeros(4), cov, size=10_000)
    advantage = Advantage()
    rows = []
    for name, logits, reward, correct, perturbation, matrix in (
        ("separated", [-0.3, -0.1, 0.1, 0.3], Reward(), True, noise, cov),
        ("near tie", [-0.003, -0.001, 0.001, 0.003], Reward(), True, noise, cov),
        ("identical", [0, 0, 0, 0], Reward(), True, noise, cov),
        ("mixed correctness", [-0.3, -0.1, 0.1, 0.3], Reward(), [True, True, False, True], noise, cov),
        ("clipping boundary", list(np.log([3.0, 3.01, 2.99, 3.02])), Reward("clipped"), True, noise, cov),
        ("threshold boundary", list(np.log([1.02, 1.03, 1.01, 1.02])), Reward("threshold"), True, noise, cov),
        ("common mode", [-0.3, -0.1, 0.1, 0.3], Reward(), True,
         np.repeat(noise[:, :1], 4, axis=1), np.full((4, 4), cov[0, 0])),
        ("outliers", [-0.3, -0.1, 0.1, 0.3], Reward(), True,
         noise * np.where(np.arange(len(noise))[:, None] % 20 == 0, 12, 1), cov * 8.15),
    ):
        rewards = reward.values(logits, correct)
        jacobian = advantage.jacobian(rewards) * reward.slope(logits, correct)[None, :]
        predicted = propagated_variance(jacobian, matrix)
        simulated = np.apply_along_axis(advantage.values, 1, reward.values(logits + perturbation, correct))
        mse = np.mean(np.sum((simulated - advantage.values(rewards))**2, axis=1))
        rows.append({"case": name, "delta_squared_error": predicted, "mc_squared_error": float(mse)})
    return rows


def _raw_identity(path):
    return sha256(path.read_bytes()).hexdigest()


def run(config, out, resume=False, recompute=False):
    out = Path(out)
    processed = out / "processed" / "results.json"
    trace_path = out / "raw" / "trace.json.gz"
    gold_path = out / "raw" / "synthetic_latent_evaluation_only.json"
    config_path = out / "raw" / "config.json"
    provenance_path = out / "raw" / "sha256.json"
    if processed.exists() and not recompute:
        saved = json.loads(processed.read_text())
        if not resume or saved["config"] != config:
            raise ValueError("Output already populated. Use --resume with the original configuration.")
        if trace_path.exists() and config_path.exists() and provenance_path.exists():
            for filename, digest in json.loads(provenance_path.read_text()).items():
                if _raw_identity(provenance_path.parent / filename) != digest:
                    raise ValueError(f"Raw diagnostic digest differs: {filename}")
            return saved

    if config["groups"] < 1 or config["blocks"] < max(config["budgets"]):
        raise ValueError("Require at least one group and enough generated blocks for the budget grid")
    if trace_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing trace differs. Choose a new output directory.")
        digests = json.loads(provenance_path.read_text())
        for filename, digest in digests.items():
            if _raw_identity(provenance_path.parent / filename) != digest:
                raise ValueError(f"Raw diagnostic digest differs: {filename}")
        trace = json.loads(gzip.decompress(trace_path.read_bytes()))
        gold = json.loads(gold_path.read_text())
    else:
        trace, gold = generate(config["seed"], config["groups"], config["blocks"])
        write_json(config_path, config)
        write_json(gold_path, gold)
        staged_trace = trace_path.with_suffix(".pending")
        staged_trace.write_bytes(gzip.compress(encoded(trace).encode(), mtime=0))
        digests = {config_path.name: _raw_identity(config_path), gold_path.name: _raw_identity(gold_path),
                   trace_path.name: _raw_identity(staged_trace)}
        write_json(provenance_path, digests)
        staged_trace.replace(trace_path)

    # Gold is bound to the scorer. The scheduling source receives only the ordered trace.
    rows = list(simulate(config, trace, gold))
    results = {"evidence": EVIDENCE, "config": config, "raw_sha256": digests,
               "runs": rows, "approximation": approximation(config["seed"])}
    write_json(processed, results)
    return results
