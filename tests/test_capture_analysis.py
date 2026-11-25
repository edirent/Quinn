"""Exercise analysis against temporary fixture captures on CPU."""
import json

import pytest

from quinn.reporting import capture_analysis
from quinn.workers import closed_capture


def capture_fixture(directory, mode, session, count):
    def candidate(i):
        return {"candidate": f"L1-001/v{i}", "task": "L1-001", "variant": f"v{i}",
                "category": "matrix multiplication", "split": "development",
                "candidate_recipe_sha256": f"cpu-fixture-{i}"}

    rows = [candidate(i) for i in range(4)]
    directory.mkdir()
    opts = {"condition": "operator-declared-isolated", "session": session, "mode": mode}
    run = {"type": "a100_diagnostic_capture", "options": opts, "corpus": rows,
           "packages": {"torch": "fixture"}}
    (directory / "run.json").write_text(json.dumps(run))
    completed = []
    for i, item in enumerate(rows):
        path = directory / "jobs" / item["task"] / item["variant"] / "1.jsonl"
        path.parent.mkdir(parents=True)
        start = {"record": "start", "spec": {"gpu_uuid": "TEST-CPU-FIXTURE",
                                             "fixed_input_seed": 0}}
        valid = {"record": "validation", "correctness_status": "passed", "compilation_status": "passed"}
        blocks = []
        for k in range(count):
            rival = .7 + i * .1 + (k % 3)*.001
            blocks.append({"record": "block", "block": k, "block_wall_ms": 2.5,
                           "samples": [{"role": "R", "ms": 1.1}, {"role": "C", "ms": rival}],
                           "log_speedup": 0.5 - i * 0.15 + (k % 3) * .001})
        path.write_text("".join(json.dumps(row) + "\n" for row in
                                [start, valid, *blocks, {"record": "complete"}]))
        assert closed_capture(path)
        completed.append(str(path.relative_to(directory)))
    (directory / "accounting.json").write_text(json.dumps({"completed_artifacts": completed}))


def test_physical_capture_tooling_fixture(tmp_path):
    dirs = [tmp_path / x for x in ["reference-a", "reference-b", "paired"]]
    for path, method, blocks in zip(dirs, ["gold", "gold", "m2"], [200, 200, 12]):
        capture_fixture(path, method, path.name, blocks)
    out = tmp_path / "analysis"
    summary = capture_analysis(dirs, out, replay_budgets=[60], block_bound=3)
    assert summary["qualified_reference_candidates"] == 4
    assert summary["comparison_groups"] == 1
    assert summary["physical_replay_rows"] == 4
    assert (out / "fig01_runtime_variability.pdf").read_bytes().startswith(b"%PDF")
    assert "TEST-CPU-FIXTURE" in (out / "physical_replay_metrics.csv").read_text()
    with pytest.raises(ValueError, match="exceeds declared replay bound"):
        capture_analysis(dirs, out, replay_budgets=[60], block_bound=2)


def test_partial_artifact_is_not_complete(tmp_path):
    raw = tmp_path / "1.jsonl"
    assert not closed_capture(raw)
    raw.write_text('{"record": "start"}\n{"record"')
    assert not closed_capture(raw)
