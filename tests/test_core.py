from dataclasses import replace
import concurrent.futures
import json
from pathlib import Path
import random

import numpy as np
import pytest

from quinn.advantages import Advantage, propagated_variance
from quinn.diagnostics import generate, simulate, approximation
from quinn.estimation import Observation, Reward, estimate, long_run_variance
from quinn.measurement import cuda_event_ms, measure_block, sequence
from quinn.scheduling import (Candidate, Policy, Quote, marginal_values,
                              residual_covariance, schedule, uncertainty_model)
from quinn.scheduling.replay import Replay
from quinn.telemetry import a100_devices
from quinn.workers.budget import Ledger
from quinn.workloads import manifest


@pytest.mark.parametrize("mode", ["grpo", "rloo", "centered"])
@pytest.mark.parametrize("values", [[1.0, 0.5, 0.2, -1.0], [0, 0, 0, 0], [-0.1, 0.2, -2, 0.3]])
def test_mask_and_jacobian(mode, values):
    advantage = Advantage(mode)
    mask = [True, False, True, True]
    point = np.asarray(values, dtype=float)
    deltas = np.eye(4) * 1e-8
    numerical = np.column_stack([(advantage.values(point+d, mask)
                                  - advantage.values(point-d, mask)) / 2e-8 for d in deltas])
    np.testing.assert_allclose(advantage.jacobian(point, mask), numerical, rtol=1e-6, atol=1e-8)
    assert advantage.values(point, mask)[1] == 0


def test_grpo_upstream_conventions_and_rloo():
    values = np.array([0.2, -0.5, 0.8, 0.7])
    a = Advantage()
    np.testing.assert_allclose(a.values(values), (values-values.mean())/(values.std(ddof=1)+1e-6))
    np.testing.assert_allclose(a.values([0.1]), [0.1/(1+1e-6)])
    np.testing.assert_allclose(a.values([0, 1e-9]), [-0.5e-9/(1+1e-6), 0.5e-9/(1+1e-6)])
    np.testing.assert_allclose(Advantage("rloo").values(values),
                               [x - np.delete(values, i).mean() for i, x in enumerate(values)])
    assert propagated_variance(a.jacobian(values), np.ones((4, 4))) == pytest.approx(0, abs=1e-14)


def test_reward_and_estimator():
    x = np.log([0.5, 2, 5])
    for kind, expected in (("log", x), ("raw", [0.5, 2., 5.]), ("clipped", [0.5, 2, 3]),
                           ("threshold", [0, 1, 1]), ("kernelgym", [0.75, 1.5, 2.])):
        reward = Reward(kind, correctness_weight=.5, performance_weight=.5)
        np.testing.assert_allclose(reward.values(x), expected)
        missing = estimate("bad", "task", False, [], reward)
        assert missing.reward_mean == reward.penalty and missing.measurement_count == 0
    points = [Observation(i, 0.02 * i, 3, 3.1) for i in range(8)]
    result = estimate("ok", "task", True, points, Reward())
    assert result.reward_mean == pytest.approx(0.07)
    assert result.gpu_occupied_ms == 24 and result.measurement_count == 8
    assert result.lower_bound < result.reward_mean < result.upper_bound
    assert long_run_variance(range(30)) > np.var(range(30), ddof=1)


def test_paired_measurement_and_event_interface():
    calls = []
    class FakeEvent:
        def __init__(self, enable_timing):
            assert enable_timing

        def record(self):
            calls.append("event")

        def elapsed_time(self, end):
            assert isinstance(end, FakeEvent)
            calls.append("elapsed")
            return 6.0

    class FakeCuda:
        Event = FakeEvent

        @staticmethod
        def synchronize(device):
            calls.append(device)

    class FakeTorch:
        cuda = FakeCuda

    assert cuda_event_ms(FakeTorch, "gpu", lambda: calls.append("launch"), launches=3) == 2
    assert calls == ["event", "launch", "launch", "launch", "event", "gpu", "elapsed"]
    data = measure_block("RCCR", {"R": 2, "C": 1}.__getitem__)
    assert data["event_sum_ms"] == 6
    assert data["log_speedup"] == pytest.approx(np.log(2))
    assert sequence("m2", random.Random(1), "RCRC") == "RCRC"
    assert {sequence("m2", random.Random(i), "random") for i in range(20)} == {"RCCR", "CRRC"}
    with pytest.raises(ValueError):
        measure_block("RCCR", lambda _: 0)


def fixture_source():
    candidates = [Candidate("a", "g", True), Candidate("b", "g", True), Candidate("bad", "g", False)]
    streams = {c.key: [{"epoch": i, "log_speedup": (0.1 if c.key == "a" else 0.4) + (-1)**i*0.01,
                         "gpu_occupied_ms": 1, "wall_time_ms": 2} for i in range(10)]
               for c in candidates if c.correct}
    return candidates, Replay(streams, {"a": [1, 2], "b": [1, 2]})


def test_scheduler_replay_no_future_and_invalid_handling():
    candidates, source = fixture_source()
    with pytest.raises(ValueError, match="next observation"):
        source.acquire(candidates[0], 4)
    policy = Policy(initial=2)
    first = schedule(candidates, source, 6, 12, policy)
    second = schedule(*fixture_source(), 6, 12, policy)
    assert first["events"] == second["events"]
    assert first["gpu_ms"] == 6 and first["wall_ms"] == 12 and not first["incomplete_groups"]
    assert first["estimates"]["bad"].reward_mean == Reward().penalty
    assert "bad" not in first["events"]
    assert not hasattr(source, "gold")


def test_budget_deadline_initialization_and_stopping():
    policy = Policy(initial=2)
    budget = schedule(*fixture_source(), 5, 100, policy)
    assert len(budget["events"]) == 5
    deadline = schedule(*fixture_source(), 100, 4, policy)
    assert len(deadline["events"]) == 2 and deadline["incomplete_groups"] == ["g"]
    stopped = schedule(*fixture_source(), 100, 100, replace(policy, stop_rmse=100))
    assert len(stopped["events"]) == 4
    with pytest.raises(RuntimeError, match="admission contract"):
        class IncorrectQuote:
            def quote(self, candidate, index):
                return Quote(1, 1)

            def acquire(self, candidate, index):
                return Observation(index, 0, 2, 2)

        schedule(fixture_source()[0], IncorrectQuote(), 1, 1, policy)


def test_covariance_and_priority():
    offsets = np.array([0.0, 0.1, 0.4, 0.8])
    noise = np.array([0.0, 0.02, 0.03, -0.016, -0.024])
    histories = [[Observation(i, float(x + offset), 1, 1) for i, x in enumerate(noise)]
                 for offset in offsets]
    policy = Policy()
    correct = [True] * 4
    _, covariance, counts = uncertainty_model(histories, np.asarray(correct), policy)
    assert covariance[0, 1] > 0
    reduced = residual_covariance(covariance, 0, counts[0])
    assert np.linalg.eigvalsh(reduced).min() > 0
    assert reduced[0, 0] == pytest.approx(covariance[0, 0] * 5/6)
    priorities, risk = marginal_values(histories, correct, policy, Reward(), Advantage())
    rewards, _ = marginal_values(histories, correct, replace(policy, objective="reward"),
                                 Reward(), Advantage())
    assert risk > 0 and not np.allclose(priorities, rewards)
    assert np.ptp(priorities) > 0 and np.ptp(rewards) < 1e-12


def test_synthetic_coverage_of_failure_modes():
    trace, truth = generate(1, 2, 10)
    assert trace["evidence"] == "synthetic_cpu_diagnostic"
    rows = list(simulate({"reward": "log", "budgets": [4], "initial": 2,
                          "deadline_scale": 1.15}, trace, truth))
    assert len(rows) == 16 and all(row["complete"] for row in rows)
    data = {row["case"]: row for row in approximation(1)}
    assert data["threshold boundary"]["delta_squared_error"] == 0
    assert data["threshold boundary"]["mc_squared_error"] > .1
    assert data["common mode"]["mc_squared_error"] < 1e-15


def test_budget_ledger_concurrency_and_recovery(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite", 10)
    def admit(job):
        try:
            return ledger.reserve(job, "reward_evaluation", 6)
        except ValueError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        attempts = list(executor.map(admit, ("one", "two")))
    assert sum(attempt is not None for attempt in attempts) == 1
    ledger.recover(("one", "two")[attempts.index(next(x for x in attempts if x))])
    assert ledger.summary()["charged_gpu_hours"] == pytest.approx(6/3600)
    retry = ledger.reserve("three", "reward_evaluation", 4)
    ledger.finish(retry, 2, True, "complete")
    assert ledger.reserve("three", "reward_evaluation", 4, resume=True) is None
    with pytest.raises(ValueError, match="Completed"):
        ledger.reserve("three", "reward_evaluation", 4)
    assert ledger.summary()["charged_gpu_hours"] == pytest.approx(8/3600)


def test_hardware_admission(monkeypatch):
    monkeypatch.setattr("quinn.telemetry.inventory",
                        lambda: [{"index": "0", "name": "RTX 4090"}])
    with pytest.raises(ValueError, match="not an accessible A100"):
        a100_devices(["0"])
    with pytest.raises(ValueError, match="distinct"):
        a100_devices(["0", "0"])


def test_pinned_manifest_if_fetched():
    root = Path(".upstream/kernelbench")
    if not (root / ".git").exists():
        pytest.skip("make fetch-upstream to test the pinned corpus")
    rows = manifest(root)
    assert len(rows) == 180
    assert len({r["task"] for r in rows}) == 30
    assert len({r["candidate_recipe_sha256"] for r in rows}) == 180
    assert {r["split"] for r in rows} == {"development", "held_out"}
    assert all(r["rollout_group"] is None for r in rows)


@pytest.mark.a100
def test_physical_smoke(tmp_path):
    from quinn.telemetry import inventory
    gpu = next((item for item in inventory() if "A100" in item["name"]), None)
    if gpu is None:
        pytest.skip("physical A100 absent")
    pytest.importorskip("torch")
    import yaml
    from quinn.workers import capture
    if not Path(".upstream/kernelbench/.git").exists():
        pytest.skip("make fetch-upstream to test the pinned corpus")
    config = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    config.pop("kind")
    config["ledger"] = str(tmp_path / "ledger.sqlite")
    run = tmp_path / "capture"
    capture(config, [gpu["index"]], 0.1, run, False)
    records = [json.loads(line) for line in next(run.glob("jobs/*/*/*.jsonl")).read_text().splitlines()]
    blocks = [record for record in records if record["record"] == "block"]
    assert len(blocks) == config["blocks"]
    assert all({item["role"] for item in block["samples"]} == {"R", "C"} for block in blocks)
    result = capture(config, [gpu["index"]], 0.1, run, True)
    assert result["worker_summary"][0]["skipped"] == 1


def test_recovery_adopts_completed_raw_without_a_second_job(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite", 10)
    attempt = ledger.reserve("completed-before-coordinator-crash", "reward_evaluation", 8)
    ledger.recover("completed-before-coordinator-crash", {attempt})
    assert ledger.reserve("completed-before-coordinator-crash", "reward_evaluation", 2, resume=True) is None
    assert ledger.summary()["charged_gpu_hours"] == pytest.approx(8 / 3600)


def test_offline_categorical_policy_update():
    from quinn.training.offline import update_metrics
    features = [[1.0, 0], [0, 1.0], [-1.0, 0]]
    oracle = update_metrics(features, [1.0, 0, -1.0], [1.0, 0, -1.0])
    assert oracle["gradient_cosine"] == pytest.approx(1)
    assert oracle["one_step_gain_shortfall"] == 0
    assert oracle["update_true_reward_gain"] > 0
    wrong = update_metrics(features, [1.0, 0, -1.0], [-1.0, 0, 1.0])
    assert wrong["gradient_cosine"] == pytest.approx(-1)
    assert wrong["update_true_reward_gain"] < 0
