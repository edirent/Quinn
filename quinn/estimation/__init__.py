"""Reward transforms and block-level inference."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import t


@dataclass(frozen=True)
class Reward:
    kind: str = "log"
    cap: float = 3.0
    threshold: float = 1.02
    penalty: float = -1.0
    correctness_weight: float = 0.5
    performance_weight: float = 0.5

    def __post_init__(self):
        if self.kind not in {"log", "raw", "clipped", "threshold", "kernelgym"}:
            raise ValueError("Unknown reward transformation")
        if self.cap <= 0 or self.threshold <= 0:
            raise ValueError("cap and threshold must be positive")

    def values(self, log_speedup, correct=True):
        z = np.asarray(log_speedup)
        speedup = np.exp(np.clip(z, -700, 700))
        if self.kind == "log":
            value = z
        elif self.kind == "raw":
            value = speedup
        elif self.kind == "threshold":
            value = (z >= np.log(self.threshold)).astype(float)
        else:
            value = np.minimum(speedup, self.cap)
            if self.kind == "kernelgym":
                value = self.correctness_weight + self.performance_weight * value
        return np.where(correct, value, self.penalty)

    def slope(self, z, correct=True):
        z = np.asarray(z, dtype=float)
        if self.kind == "log":
            slope = np.ones_like(z)
        elif self.kind == "threshold":
            slope = np.zeros_like(z)
        else:
            slope = np.exp(np.clip(z, -700, 700))
            if self.kind != "raw":
                slope = np.where(z < np.log(self.cap), slope, 0)
            if self.kind == "kernelgym":
                slope = slope * self.performance_weight
        return np.where(correct, slope, 0)


@dataclass(frozen=True)
class Observation:
    epoch: int
    log_speedup: float
    gpu_occupied_ms: float
    wall_time_ms: float
    telemetry: dict | None = None


@dataclass(frozen=True)
class RewardEstimate:
    candidate_id: str
    group_id: str
    correctness: bool
    reward_mean: float
    reward_variance: float
    lower_bound: float
    upper_bound: float
    measurement_count: int
    gpu_occupied_ms: float
    wall_time_ms: float
    telemetry: dict


def location(values, estimator):
    values = np.asarray(values, dtype=float)
    if estimator == "median":
        return float(np.median(values))
    if estimator == "mom":
        return float(np.median([x.mean() for x in np.array_split(values, min(5, len(values)))]))
    if estimator == "trimmed":
        trim = len(values) // 10
        values = np.sort(values)[trim:len(values) - trim]
    elif estimator != "mean":
        raise ValueError("unknown location estimator")
    return float(values.mean())


def long_run_variance(values, floor=1e-6):
    """Bartlett HAC estimate over ordered blocks, with a conservative IID floor."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return floor
    centered = values - values.mean()
    n = len(values)
    variance = np.var(values, ddof=1)
    lags = min(n - 1, int(n ** (1 / 3)))
    hac = variance
    for lag in range(1, lags + 1):
        hac += 2 * (1 - lag / (lags + 1)) * np.dot(centered[lag:], centered[:-lag]) / n
    return float(max(hac, variance, floor))


def estimate(candidate_id, group_id, correct, observations, reward, estimator="mean"):
    if not correct:
        return RewardEstimate(candidate_id, group_id, False, reward.penalty, 0.0,
                              reward.penalty, reward.penalty, 0, 0, 0, {})
    if not observations:
        raise ValueError(f"Missing measurements for {candidate_id}")
    z = np.array([x.log_speedup for x in observations])
    mean = location(z, estimator)
    variance = long_run_variance(z) / len(z)
    half_width = t.ppf(0.975, max(1, len(z) - 1)) * np.sqrt(variance)
    point = float(reward.values(mean))
    if reward.kind == "threshold":
        probability = t.cdf((mean - np.log(reward.threshold)) / np.sqrt(variance),
                            max(1, len(z) - 1))
        reward_variance = probability * (1 - probability)
    else:
        reward_variance = float(reward.slope(mean) ** 2 * variance)
    low, high = reward.values([mean - half_width, mean + half_width])
    return RewardEstimate(candidate_id, group_id, True, point,
                          float(reward_variance), float(low), float(high),
                          len(z), sum(x.gpu_occupied_ms for x in observations),
                          sum(x.wall_time_ms for x in observations),
                          observations[-1].telemetry or {})
