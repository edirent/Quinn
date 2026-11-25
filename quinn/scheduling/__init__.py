"""Greedy allocation over observed blocks and public admission bounds."""
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
from scipy.special import ndtr

from quinn.advantages import Advantage
from quinn.estimation import Observation, Reward, estimate, location, long_run_variance


@dataclass(frozen=True)
class Candidate:
    key: str
    group: str
    correct: bool


@dataclass(frozen=True)
class Quote:
    gpu_ms: float
    wall_ms: float


class BlockSource(Protocol):
    def quote(self, candidate: Candidate, index: int) -> Quote | None: ...
    def acquire(self, candidate: Candidate, index: int) -> Observation: ...


@dataclass(frozen=True)
class Policy:
    objective: str = "advantage"
    initial: int = 4
    fixed_blocks: int | None = None
    covariance: bool = True
    cost_aware: bool = True
    stop_rmse: float | None = None
    estimator: str = "mean"
    nonlinear: bool = True
    variance_floor: float = 1e-6

    def __post_init__(self):
        if self.objective not in {"fixed", "reward", "rank", "advantage"}:
            raise ValueError("unknown scheduling objective")
        if self.initial < 2 or self.variance_floor <= 0:
            raise ValueError("need two initial blocks and a positive variance floor")
        if self.fixed_blocks is not None and self.fixed_blocks < self.initial:
            raise ValueError("fixed cap is below initialization")


def uncertainty_model(histories, correct, policy):
    width = len(histories)
    z = np.zeros(width)
    variances = np.zeros(width)
    counts = np.array([len(history) for history in histories])
    active = np.flatnonzero(correct & (counts > 0))
    for i in active:
        values = [sample.log_speedup for sample in histories[i]]
        z[i] = location(values, policy.estimator)
        variances[i] = long_run_variance(values, policy.variance_floor) / counts[i]
    covariance = np.diag(variances)
    if policy.covariance and len(active) > 1:
        epochs = [{x.epoch: x.log_speedup for x in histories[i]} for i in active]
        common = sorted(set.intersection(*(set(item) for item in epochs)))
        if len(common) >= 4:
            matched = np.array([[item[epoch] for epoch in common] for item in epochs])
            centered = matched - matched.mean(axis=1, keepdims=True)
            norm = np.linalg.norm(centered, axis=1, keepdims=True)
            normalized = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
            scale = np.sqrt(variances[active] * len(common) / counts[active])
            correlated = (normalized @ normalized.T) * np.outer(scale, scale)
            correlated = 0.5 * correlated
            np.fill_diagonal(correlated, variances[active])
            covariance[np.ix_(active, active)] = correlated
    return z, covariance, counts


def residual_covariance(covariance, index, count):
    column = covariance[:, index]
    denominator = (count + 1) * covariance[index, index]
    return covariance - np.outer(column, column) / denominator


def _sampling_risk(draws, point, covariance, transform):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    perturbation = (draws * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    return float(np.mean(np.sum((transform(point + perturbation) - transform(point)) ** 2, axis=-1)))


def marginal_values(histories, correct, policy, reward, advantage):
    correct = np.asarray(correct, dtype=bool)
    z, covariance, counts = uncertainty_model(histories, correct, policy)
    scores = np.zeros(len(histories))
    if not np.any(correct):
        return scores, 0.0
    rewards = reward.values(z, correct)
    slope = reward.slope(z, correct)
    gradient = advantage.jacobian(rewards) * slope[None, :]
    active = np.flatnonzero(correct & (counts > 0))
    centered = rewards - rewards.mean()
    spread = np.linalg.norm(centered)
    linear_noise = np.linalg.norm(slope * np.sqrt(np.diag(covariance)))
    use_mc = policy.nonlinear and (
        reward.kind in {"threshold", "clipped", "kernelgym"}
        or (policy.objective == "advantage" and spread <= linear_noise)
    )
    if policy.objective == "rank":
        # Sum posterior sign-error probabilities over uncertain log-speedup orderings.
        a, b = np.triu_indices(len(active), 1)
        left, right = active[a], active[b]
        gap = np.abs(z[left] - z[right])

        def risk(matrix):
            variance = matrix[left, left] + matrix[right, right] - 2 * matrix[left, right]
            return float(np.sum(ndtr(-gap / np.sqrt(np.maximum(variance, 1e-18)))))
    elif use_mc:
        draws = np.random.default_rng(179).standard_normal((128, len(z)))

        def transform(point):
            values = reward.values(point, correct)
            if policy.objective == "advantage":
                return np.apply_along_axis(advantage.values, -1, values)
            return values

        def risk(matrix):
            return _sampling_risk(draws, z, matrix, transform)
    else:
        if policy.objective == "reward":
            gradient = np.diag(slope)

        def risk(matrix):
            return float(np.trace(gradient @ matrix @ gradient.T))

    current = max(0.0, risk(covariance))
    for i in active:
        scores[i] = max(0.0, current - risk(residual_covariance(covariance, i, counts[i])))
    return scores, current


def schedule(candidates, source, budget_ms, deadline_ms, policy=Policy(), reward=Reward(),
             advantage=Advantage()):
    """Sequential policy, independent of the origin of the admitted observations."""
    if budget_ms < 0 or deadline_ms < 0:
        raise ValueError("negative budget or deadline")
    if len({candidate.key for candidate in candidates}) != len(candidates):
        raise ValueError("candidate keys must be globally unique")
    grouped = {}
    history = {c.key: [] for c in candidates}
    for c in candidates:
        grouped.setdefault(c.group, []).append(c)
    gpu_ms = wall_ms = cpu_ms = 0.0
    events = []
    stop = "no_admissible_action"

    while True:
        start = perf_counter()
        options = []
        for group in grouped.values():
            histories = [history[c.key] for c in group]
            warm = any(c.correct and len(h) < policy.initial for c, h in zip(group, histories))
            if warm or policy.objective == "fixed":
                scores = [-len(h) for h in histories]
            else:
                scores, risk = marginal_values(histories, [c.correct for c in group],
                                               policy, reward, advantage)
                if (policy.objective == "advantage" and policy.stop_rmse is not None
                        and np.sqrt(risk / len(group)) <= policy.stop_rmse):
                    continue
            for c, h, score in zip(group, histories, scores):
                index = len(h)
                if not c.correct or (warm and index >= policy.initial):
                    continue
                if policy.fixed_blocks is not None and index >= policy.fixed_blocks:
                    continue
                quote = source.quote(c, index)
                if quote is None or quote.gpu_ms > budget_ms - gpu_ms:
                    continue
                if quote.gpu_ms <= 0 or quote.wall_ms <= 0:
                    raise ValueError("admission bounds must be positive")
                if quote.wall_ms > deadline_ms - wall_ms:
                    continue
                expected_cost = np.mean([x.gpu_occupied_ms for x in h]) if h else quote.gpu_ms
                if not warm and policy.cost_aware and policy.objective != "fixed":
                    score /= max(expected_cost, 1e-12)
                options.append((warm, score, c, index, quote))
        cpu_ms += (perf_counter() - start) * 1000
        if not options:
            break
        _, _, candidate, index, quote = max(options, key=lambda action: action[:2])
        sample = source.acquire(candidate, index)
        if not (0 <= sample.gpu_occupied_ms <= quote.gpu_ms
                and 0 <= sample.wall_time_ms <= quote.wall_ms
                and np.isfinite(sample.log_speedup)):
            raise RuntimeError("block source violated admission contract")
        gpu_ms += sample.gpu_occupied_ms
        wall_ms += sample.wall_time_ms
        history[candidate.key].append(sample)
        events.append(candidate.key)

    estimates = {}
    for c in candidates:
        samples = history[c.key]
        if samples or not c.correct:
            estimates[c.key] = estimate(c.key, c.group, c.correct, samples,
                                        reward, policy.estimator)
    return {"estimates": estimates, "events": events, "gpu_ms": gpu_ms,
            "wall_ms": wall_ms, "scheduler_cpu_ms": cpu_ms, "stop": stop,
            "incomplete_groups": sorted({c.group for c in candidates
                                         if c.correct and len(history[c.key]) < policy.initial})}
