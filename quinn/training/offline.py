"""A synthetic one-step score-function diagnostic on a known categorical policy."""
import numpy as np
from scipy.special import softmax

from quinn.advantages import Advantage
from quinn.estimation import Reward


def update_metrics(features, exact_rewards, estimated_rewards, learning_rate=0.15):
    """Evaluate policy gradients at uniform action probabilities and a single update."""
    features = np.asarray(features)
    exact_rewards, estimated_rewards = np.asarray(exact_rewards), np.asarray(estimated_rewards)
    score_features = features - features.mean(axis=0)
    normalization = Advantage()

    def gradient(rewards):
        return score_features.T @ normalization.values(rewards) / len(rewards)

    def expected_after(step):
        return float(np.dot(softmax(features @ (learning_rate * step)), exact_rewards))

    estimate, exact = gradient(estimated_rewards), gradient(exact_rewards)
    denominator = np.linalg.norm(estimate) * np.linalg.norm(exact)
    initial = float(exact_rewards.mean())
    optimal = expected_after(exact)
    observed = expected_after(estimate)
    return {"gradient_l2_error": float(np.linalg.norm(estimate - exact)),
            "gradient_cosine": float(np.dot(estimate, exact) / denominator) if denominator else None,
            "update_true_reward_gain": observed - initial,
            "reference_update_true_reward_gain": optimal - initial,
            "one_step_gain_shortfall": optimal - observed}


def offline_policy_rows(data, trace, gold, policies, methods):
    """Rebuild final rewards from each policy's in-order allocation, then score the update."""
    reward = Reward(data["config"]["reward"])
    by_group = {}
    rng = np.random.default_rng(data["config"]["seed"] + 8103)
    for group_id, group in sorted(trace["groups"].items()):
        correct = np.array([candidate["correct"] for candidate in group["candidates"]])
        exact = reward.values(gold[group_id], correct)
        by_group[group_id] = (correct, exact, rng.normal(size=(len(correct), 4)))
    for row in data["runs"]:
        if not row["complete"] or row["method"] not in methods:
            continue
        group_id, method = row["group"], row["method"]
        group = trace["groups"][group_id]
        correct, exact, features = by_group[group_id]
        source = group[policies[method][0]]
        sampled = iter(row["counts"])
        observations = [
            np.mean([entry["log_speedup"] for entry in source[candidate["key"]][:next(sampled)]])
            if valid else 0
            for candidate, valid in zip(group["candidates"], correct)
        ]
        estimates = reward.values(observations, correct)
        yield {"group": group_id, "method": method, "budget_level": row["budget_level"],
               **update_metrics(features, exact, estimates)}
