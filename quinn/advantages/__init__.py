"""Group-level terminal outcome advantages and their derivatives."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Advantage:
    mode: str = "grpo"
    epsilon: float = 1e-6
    ddof: int = 1
    small_std_fallback: bool = True

    def __post_init__(self):
        if self.mode not in {"grpo", "centered", "rloo"} or self.ddof not in (0, 1):
            raise ValueError("Unknown advantage convention")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    def values(self, rewards, mask=None):
        return self._apply(rewards, mask, derivative=False)

    def jacobian(self, rewards, mask=None):
        return self._apply(rewards, mask, derivative=True)

    def _apply(self, rewards, mask, derivative):
        rewards = np.asarray(rewards, dtype=float)
        indices = np.flatnonzero(np.ones(len(rewards), bool) if mask is None else mask)
        n = len(indices)
        values = rewards[indices]
        result = np.zeros((len(rewards), len(rewards)) if derivative else len(rewards))
        if n == 0:
            return result
        if n == 1:
            scale = 1 / (1 + self.epsilon) if self.mode == "grpo" else 1
            values = values * scale
            jacobian = np.eye(1) * scale
        else:
            projector = np.eye(n) - 1 / n
            centered = values - values.mean()
            degrees = n - self.ddof
            std = np.linalg.norm(centered) / np.sqrt(degrees)
            if self.mode == "grpo":
                fallback = self.small_std_fallback and std < self.epsilon
                denominator = (1 if fallback else std) + self.epsilon
                values = centered / denominator
                jacobian = projector / denominator
                if std > 0 and not fallback:
                    jacobian -= np.outer(centered, centered) / (
                        degrees * std * denominator**2
                    )
            else:
                scale = n / (n - 1) if self.mode == "rloo" else 1
                values = centered * scale
                jacobian = projector * scale
        if derivative:
            result[np.ix_(indices, indices)] = jacobian
        else:
            result[indices] = values
        return result


def propagated_variance(jacobian, covariance):
    return float(np.maximum(0, np.trace(jacobian @ covariance @ jacobian.T)))
