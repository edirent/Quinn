# Quinn

Quinn studies whether adjusting the GPU measurement budget according to its expected effect on group-relative advantages improves reinforcement-learning reward fidelity. This repository contains a pinned KernelBench diagnostic panel, an A100 capture pipeline, and a scheduler evaluated on reproducible CPU-generated traces. The A100 pilot is blocked in the recorded environment. Online RL is gated.

## Reproduce the CPU diagnostic

Use Linux, Python 3.12, uv, Git, and GNU Make from the repository root.

```sh
make setup
make reproduce
```

This runs tests, records hardware, fetches two pinned upstream revisions, and rebuilds the summaries from the compressed raw diagnostic trace. The raw trace and latent scoring data have separate SHA-256 entries. Measured CPU elapsed time varies by host and load.

- [Experimental report](docs/final_experimental_report.md)
- [Raw and processed CPU artifacts](artifacts/diagnostics/)
- [Offline policy-step diagnostic](artifacts/diagnostics/processed/offline_policy_summary.json)
- [Methodology](docs/methodology.md)
- [Advantage derivation](docs/mathematical_formulation.md)

## A100 setup and capture

Use this checkout on a Linux host with one or two physical A100s and a working CUDA driver. Device indices are the physical IDs from `nvidia-smi`.

```sh
make setup setup-gpu hardware-check corpus
make smoke-test GPUS=0
make pilot GPUS=0,1
```

The capture requires a full A100 and validates the device again in the child process. View the [experimental protocol](docs/experimental_protocol.md) for independent reference sessions, baseline budgets, controlled contention, telemetry checks, and offline analysis.

```sh
.venv/bin/python -m quinn.run --config configs/evaluation.yaml \
  --gpus 0,1 --gpu-hour-budget 96 --resume
```

The default shared SQLite ledger uses a 96 aggregate GPU-hour ceiling. Use the same ledger across phases and sessions. Raw capture paths include the session name, while worker logs and measurements include job identifiers. Resuming requires an identical immutable configuration. Use a new session and output directory for changed settings. `make evaluate-system` performs a fixed two-worker-capable diagnostic capture with service accounting. `make train-rl` checks the training gate.
