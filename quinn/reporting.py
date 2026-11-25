"""Evidence-labeled summaries generated from raw measured or synthetic artifacts."""
from collections import defaultdict
import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quinn.advantages import Advantage
from quinn.diagnostics import EVIDENCE, POLICIES, fidelity
from quinn.estimation import Reward, estimate, Observation
from quinn.scheduling import Candidate, schedule
from quinn.scheduling.replay import Replay
from quinn.training.offline import offline_policy_rows
from quinn.workers import write_json

METHODS = ("B3 Fixed paired", "B4 Reward variance", "B5 Rank", "B6 Quinn")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKS = ("o", "^", "s", "D")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "legend.frameon": False})


def bootstrap(values, rng, repetitions=3000):
    values = np.asarray(values)
    if not len(values):
        return None, None, None
    center = float(values.mean())
    if len(values) == 1:
        return center, None, None
    resampled = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return center, *map(float, np.quantile(resampled, [0.025, 0.975]))


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _summary(rows, budgets):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"], row["budget_level"]].append(row)
    rng = np.random.default_rng(18056)
    summary = []
    for budget in budgets:
        for method in dict.fromkeys(row["method"] for row in rows):
            cohort = grouped[method, budget]
            completed = [row for row in cohort if row["complete"]]
            fixed = {row["group"]: row for row in grouped[METHODS[0], budget] if row["complete"]}
            quinn = {row["group"]: row for row in grouped["B6 Quinn", budget] if row["complete"]}
            row = {"method": method, "budget_level": budget, "groups": len(cohort),
                   "complete_groups": len(completed)}
            for metric in ("advantage_rmse", "sign_error", "reward_rmse", "coverage", "gpu_ms",
                           "wall_ms", "scheduler_cpu_ms"):
                row[metric] = bootstrap([r[metric] for r in completed if r[metric] is not None], rng)
            row["delta_vs_fixed"] = bootstrap([r["advantage_rmse"] - fixed[r["group"]]["advantage_rmse"]
                                                for r in completed if r["group"] in fixed], rng)
            row["quinn_minus_method"] = bootstrap([
                quinn[r["group"]]["advantage_rmse"] - r["advantage_rmse"]
                for r in completed if r["group"] in quinn
            ], rng)
            summary.append(row)
    return summary


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def publication(root):
    datafile = root / "processed" / "results.json"
    data = json.loads(datafile.read_text())
    if data["evidence"] != EVIDENCE:
        raise ValueError("Synthetic report requires the synthetic diagnostic schema")
    budgets = data["config"]["budgets"]
    summary = _summary(data["runs"], budgets)
    figures, tables = root / "figures", root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_json(root / "processed" / "summary.json", {"evidence": EVIDENCE, "summary": summary,
                                                   "results_sha256": sha256(datafile.read_bytes()).hexdigest()})
    _csv(root / "processed" / "group_metrics.csv", data["runs"])
    for metric, ylabel, name in (
        ("advantage_rmse", "Advantage RMSE", "fig11_advantage_cost_pareto.pdf"),
        ("sign_error", "Resolved sign error fraction", "fig12_advantage_sign_inversion.pdf"),
        ("scheduler_cpu_ms", "Scheduler CPU elapsed (ms/group)", "fig14_scheduler_overhead.pdf"),
    ):
        fig, ax = plt.subplots(figsize=(3.35, 2.55), layout="constrained")
        for method, color, marker in zip(METHODS, COLORS, MARKS):
            rows = [r for r in summary if r["method"] == method and r[metric][1] is not None]
            if not rows:
                continue
            x = [r["gpu_ms"][0] for r in rows]
            center, lo, hi = np.asarray([r[metric] for r in rows]).T
            ax.errorbar(x, center, yerr=np.array([center - lo, hi - center]), color=color,
                        marker=marker, linewidth=1.1, markersize=3, capsize=2, label=method)
        ax.set(xlabel="Mean modeled occupancy (ms/group)", ylabel=ylabel, title="Synthetic CPU diagnostic")
        ax.legend(fontsize=6.5)
        _save(fig, figures / name)

    fig, ax = plt.subplots(figsize=(3.35, 2.55), layout="constrained")
    cases = data["approximation"]
    indices = np.arange(len(cases))
    floor = 1e-8
    ax.scatter(indices - 0.11, [max(x["delta_squared_error"], floor) for x in cases],
               c=COLORS[0], label="First order")
    ax.scatter(indices + 0.11, [max(x["mc_squared_error"], floor) for x in cases],
               c=COLORS[1], marker="^", label="Monte Carlo")
    ax.set(yscale="log", ylabel="Expected squared advantage error",
           title="Synthetic CPU diagnostic", xticks=indices)
    ax.set_xticklabels([x["case"] for x in cases], rotation=45, ha="right")
    ax.legend(fontsize=7, loc="upper left")
    fig.text(.99, .005, "Display floor at $10^{-8}$", ha="right", fontsize=6)
    _save(fig, figures / "fig_delta_approximation.pdf")

    mid = budgets[len(budgets) // 2]
    fig, ax = plt.subplots(figsize=(3.35, 2.4), layout="constrained")
    allocation = [[count for row in data["runs"] if row["method"] == method
                   and row["budget_level"] == mid and row["complete"] for count in row["counts"]]
                  for method in METHODS]
    if all(allocation):
        ax.boxplot(allocation, showfliers=True)
        ax.set_xticks(np.arange(1, len(METHODS) + 1), ["B3", "B4", "B5", "B6"])
    ax.set(xlabel="Scheduling method", ylabel="Blocks per correct candidate",
           title="Synthetic CPU diagnostic")
    _save(fig, figures / "fig13_measurement_allocation.pdf")
    for name, show in (("scheduler_results", METHODS),
                       ("scheduler_ablation", ("B6 Quinn", "Quinn-NoPairing", "B4 Reward variance",
                                               "Quinn-NoCostAwareness", "Quinn-NoCovariance",
                                               "Quinn-NoDeadline", "B3 Fixed paired"))):
        lines = [r"\begin{tabular}{lrr}", r"\hline",
                 r"Method & Modeled cost (ms) & Advantage RMSE [95\% interval] \\", r"\hline"]
        for row in summary:
            if row["budget_level"] != mid or row["method"] not in show:
                continue
            point, lo, hi = row["advantage_rmse"]
            formatted = "unavailable" if lo is None else f"{point:.3f} [{lo:.3f}, {hi:.3f}]"
            cost = "--" if row["gpu_ms"][0] is None else f'{row["gpu_ms"][0]:.1f}'
            lines.append(f'{row["method"]} & {cost} & {formatted}' + r" \\")
        lines.extend([r"\hline", r"\end{tabular}", ""])
        (tables / f"{name}.tex").write_text("% Synthetic CPU diagnostic. No GPU measurement.\n"
                                           + "\n".join(lines))
    raw = root / "raw"
    if data.get("raw_sha256"):
        for name, digest in data["raw_sha256"].items():
            if sha256((raw / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Diagnostic raw digest differs: {name}")
        trace = json.loads(gzip.decompress((raw / "trace.json.gz").read_bytes()))
        gold = json.loads((raw / "synthetic_latent_evaluation_only.json").read_text())
        updates = list(offline_policy_rows(data, trace, gold, POLICIES, METHODS))
        _csv(root / "processed" / "offline_policy_update.csv", updates)
        rng = np.random.default_rng(61658)
        summaries = []
        for method in METHODS:
            subset = [r for r in updates if r["method"] == method and r["budget_level"] == mid]
            summaries.append({"method": method, "budget_level": mid, **{
                metric: bootstrap([r[metric] for r in subset if r[metric] is not None], rng)
                for metric in ("gradient_l2_error", "gradient_cosine", "one_step_gain_shortfall")
            }})
        write_json(root / "processed" / "offline_policy_summary.json",
                   {"evidence": "synthetic_offline_categorical_policy_step", "rows": summaries})
    return {"evidence": EVIDENCE, "groups": data["config"]["groups"], "budgets": budgets}


def _load_capture(directory):
    run = json.loads((directory / "run.json").read_text())
    if run["type"] != "a100_diagnostic_capture":
        raise ValueError("Expected a physical A100 capture")
    if not (directory / "accounting.json").exists():
        raise ValueError(f"Run has no completed coordinator summary: {directory}")
    accounting = json.loads((directory / "accounting.json").read_text())
    accepted = {directory / path for path in accounting["completed_artifacts"]}
    for item in run["corpus"]:
        base = directory / "jobs" / item["task"] / item["variant"]
        for file in sorted(base.glob("*.jsonl"), key=lambda p: int(p.stem), reverse=True):
            if file not in accepted:
                continue
            # A partial last line is possible after interruption.
            try:
                events = [json.loads(line) for line in file.read_text().splitlines()]
            except json.JSONDecodeError:
                continue
            if events and events[-1]["record"] == "complete":
                yield run, item, file, events
                break


def _case(run, item, events):
    spec = events[0]["spec"]
    valid = next(event for event in events if event["record"] == "validation")
    blocks = [event for event in events if event["record"] == "block"]
    upstream = next((event for event in events if event["record"] == "upstream_b0"), None)
    if blocks:
        values = np.asarray([[np.mean([point["ms"] for point in row["samples"] if point["role"] == role])
                              for role in "RC"] for row in blocks])
    elif any(event["record"] == "fixed_block" for event in events):
        fixed = [event for event in events if event["record"] == "fixed_block"]
        values = np.column_stack([[np.mean(event["samples_ms"]) for event in fixed
                                   if event["role"] == role] for role in "RC"])
        blocks = fixed
    elif upstream:
        values = np.column_stack([upstream["reference_ms"], upstream["candidate_ms"]])
    else:
        values = np.empty((0, 2))
    return spec, valid, blocks, values


def _block_draw(values, rng):
    n = len(values)
    length = max(1, int(np.sqrt(n)))
    starts = rng.integers(0, n, size=(n + length - 1) // length)
    indices = ((starts[:, None] + np.arange(length)) % n).ravel()[:n]
    return values[indices].mean(axis=0)


def capture_analysis(directories, output, replay_budgets=None, block_bound=None):
    output.mkdir(parents=True, exist_ok=True)
    records, references, seen = [], defaultdict(dict), set()
    for directory in directories:
        for run, item, file, events in _load_capture(directory):
            spec, valid, blocks, values = _case(run, item, events)
            opts = run["options"]
            identity = (spec["gpu_uuid"], item["candidate"], item["candidate_recipe_sha256"],
                        spec["fixed_input_seed"], run["packages"]["torch"])
            session_key = (identity, opts["session"])
            if session_key in seen:
                raise ValueError("Duplicate candidate, GPU and session combination")
            seen.add(session_key)
            records.append((identity, opts, item, file, valid, blocks, values))
            if (opts["mode"] == "gold" and valid["correctness_status"] == "passed"
                    and len(values) >= 200 and opts["condition"] == "operator-declared-isolated"):
                references[identity][opts["session"]] = values
    if not records:
        raise ValueError("No complete capture jobs found")
    rng = np.random.default_rng(20417)
    reference_stats = {}
    for identity, sessions in sorted(references.items()):
        if len(sessions) < 2:
            continue
        all_values = list(sessions.values())
        point = np.mean([v.mean(axis=0) for v in all_values], axis=0)
        array = []
        for _ in range(1000):
            indices = rng.integers(0, len(all_values), size=len(all_values))
            sample = np.mean([_block_draw(all_values[i], rng) for i in indices], axis=0)
            array.append(float(np.log(sample[0] / sample[1])))
        reference_stats[identity] = (float(np.log(point[0] / point[1])), array)
    rows, groups, replay_groups = [], defaultdict(list), defaultdict(list)
    for identity, opts, item, path, valid, blocks, values in records:
        row = {"candidate": item["candidate"], "task": item["task"], "category": item["category"],
               "split": item["split"], "gpu_uuid": identity[0], "session": opts["session"],
               "condition": opts["condition"], "protocol": opts["mode"],
               "compile_status": valid["compilation_status"],
               "correctness_status": valid["correctness_status"], "source": str(path),
               "sha256": sha256(path.read_bytes()).hexdigest(), "blocks": len(values)}
        ref = reference_stats.get(identity)
        if len(values) > 1:
            times = values[:, 1]
            cv = np.std(times, ddof=1) / times.mean()
            lag = np.corrcoef(times[:-1], times[1:])[0, 1] if np.ptp(times[:-1]) and np.ptp(times[1:]) else None
            logs = np.log(values[:, 0]) - np.log(values[:, 1])
            durations = [block["block_wall_ms"] for block in blocks]
            location = "median" if opts["mode"] == "m1" else "mean"
            if opts["mode"] in {"b0", "m0", "m1"}:
                aggregate = np.median if opts["mode"] == "m1" else np.mean
                fixed_estimate = np.log(aggregate(values[:, 0]) / aggregate(times))
            estimate_record = estimate(item["candidate"], item["task"], True,
                                       [Observation(i, float(z), 0, 0) for i, z in enumerate(logs)],
                                       Reward(), location)
            means = (float(fixed_estimate) if opts["mode"] in {"b0", "m0", "m1"}
                     else estimate_record.reward_mean)
            row.update(runtime_mean_ms=float(times.mean()), cv=float(cv),
                       p95_ms=float(np.quantile(times, 0.95)), lag1=None if lag is None else float(lag),
                       estimated_log_speedup=means, block_wall_ms=sum(durations) if durations else None)
            if ref and opts["mode"] != "gold":
                gold, samples = ref
                low, high = map(float, np.quantile(samples, [0.025, 0.975]))
                row.update(reference_log_speedup=gold, reference_ci_lower=low, reference_ci_upper=high,
                           absolute_log_error=abs(means - gold),
                           coverage=None if opts["mode"] in {"b0", "m0", "m1"} else
                           estimate_record.lower_bound <= gold <= estimate_record.upper_bound)
                group_key = (identity[0], opts["session"], item["task"], opts["mode"])
                groups[group_key].append((means, gold, samples))
                if opts["mode"] == "m2" and replay_budgets is not None:
                    replay_groups[group_key].append((item["candidate"], gold, blocks))
        rows.append(row)
    group_metrics = []
    for (uuid, session, task, mode), members in groups.items():
        expected_size = sum(row["session"] == session and row["task"] == task
                            and row["gpu_uuid"] == uuid for row in rows)
        if len(members) != expected_size or len(members) < 2:
            continue
        estimated, gold, estimates = zip(*members)
        # For confirmation, use joint session-resampled task vectors. These marginals
        # are retained solely to conservatively mark exploratory pair orders unresolved.
        lower, upper = np.quantile(estimates, [0.025, 0.975], axis=1)
        i, j = np.triu_indices(len(members), 1)
        resolved = (lower[i] > upper[j]) | (lower[j] > upper[i])
        a, b = np.asarray(estimated), np.asarray(gold)
        diag = fidelity(a, b, Advantage())
        # Interval extremes for the centered reward numerator, union-bound heuristics only.
        n = len(members)
        sign_resolved = ((n - 1) * lower > upper.sum() - upper) | ((n - 1) * upper < lower.sum() - lower)
        estimate_sign, reference_sign = np.sign(a - a.mean()), np.sign(b - b.mean())
        diag["sign_error"] = (float(np.mean(estimate_sign[sign_resolved] != reference_sign[sign_resolved]))
                              if np.any(sign_resolved) else None)
        diag["resolved_signs"] = int(sign_resolved.sum())
        diag["rank_error"] = (float(np.mean(np.sign(a[i] - a[j])[resolved] != np.sign(b[i] - b[j])[resolved]))
                              if np.any(resolved) else None)
        group_metrics.append({"gpu_uuid": uuid, "session": session, "task": task, "mode": mode,
                              "candidates": len(members), "resolved_pairs": int(resolved.sum()), **diag})
    _csv(output / "candidate_metrics.csv", rows)
    _csv(output / "group_metrics.csv", group_metrics)
    chart_rows = [row for row in rows if "cv" in row]
    if chart_rows:
        fig, ax = plt.subplots(figsize=(3.35, 2.4), layout="constrained")
        modes = sorted({row["protocol"] for row in chart_rows})
        ax.boxplot([[100 * row["cv"] for row in chart_rows if row["protocol"] == mode]
                    for mode in modes], showfliers=True)
        ax.set_xticks(np.arange(1, len(modes) + 1), modes)
        ax.set(xlabel="Capture protocol", ylabel="Per-candidate runtime CV (%)",
               title="A100 controlled diagnostic corpus")
        _save(fig, output / "fig01_runtime_variability.pdf")
    metrics = {"evidence": "a100_diagnostic_capture", "candidate_sessions": len(rows),
               "comparison_groups": len(group_metrics), "qualified_reference_candidates": len(reference_stats)}
    if replay_budgets is not None:
        replay_rows = physical_replay(replay_groups, groups, group_metrics, replay_budgets, block_bound)
        if not replay_rows:
            raise ValueError("Replay needs complete all-correct M2 groups with independent gold captures")
        _csv(output / "physical_replay_metrics.csv", replay_rows)
        metrics["physical_replay_rows"] = len(replay_rows)
        metrics["replay_evidence"] = "offline_A100_trace_reservation_proxy"
    write_json(output / "analysis.json", metrics)
    return metrics


def physical_replay(replay_groups, groups, complete_groups, budgets, block_bound):
    from dataclasses import replace

    if not np.isfinite(block_bound) or block_bound <= 0 or not all(np.isfinite(b) and b >= 1 for b in budgets):
        raise ValueError("Replay requires finite positive block bounds and budget levels")
    eligible = {(m["gpu_uuid"], m["session"], m["task"], m["mode"]) for m in complete_groups}
    output = []
    for group_key, items in sorted(replay_groups.items()):
        if group_key not in eligible or len(groups[group_key]) != len(items):
            continue
        uuid, session, task, _ = group_key
        candidates = [Candidate(key, task, True) for key, _, _ in items]
        observed = {}
        epoch = 0
        for key, _, blocks in items:
            stream = []
            for block in blocks:
                wall = block["block_wall_ms"]
                if wall > block_bound:
                    raise ValueError(f"Observed block exceeds declared replay bound: {key}")
                stream.append({"epoch": epoch, "log_speedup": block["log_speedup"],
                               "gpu_occupied_ms": wall, "wall_time_ms": wall})
                epoch += 1
            observed[key] = stream
        bounds = dict.fromkeys(observed, (block_bound, block_bound))
        reference = [g for _, g, _ in items]
        for budget in budgets:
            for method in METHODS:
                # Candidate time series were acquired serially. Their ordinal index
                # carries no cross-candidate shared reference or epoch information.
                policy = replace(POLICIES[method][1], covariance=False)
                result = schedule(candidates, Replay(observed, bounds), budget, budget, policy)
                row = {"gpu_uuid": uuid, "session": session, "task": task, "method": method,
                       "budget_ms": budget, "block_wall_charge_ms": result["gpu_ms"],
                       "scheduler_cpu_ms": result["scheduler_cpu_ms"],
                       "complete": not result["incomplete_groups"]}
                if row["complete"]:
                    estimated = [result["estimates"][c.key].reward_mean for c in candidates]
                    # Physical sign and rank metrics require joint reference draws.
                    row.update({key: value for key, value in fidelity(estimated, reference, Advantage()).items()
                                if key not in {"sign_error", "rank_error", "resolved_signs"}})
                output.append(row)
    return output
