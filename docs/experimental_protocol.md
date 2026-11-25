# Experimental protocol

Begin with an exclusive A100 allocation and record device UUIDs, driver, memory, topology, clock policy, and host conditions. Record whether the administrator has set fixed clocks. Quinn preserves clocks and records declared isolation.

## Baselines and gold

The pilot uses 64 paired blocks per candidate. Capture independently invoked warmed reference sessions on the same physical devices with different session IDs and RNG seeds. The timed inputs stay fixed by task.

```sh
RUN=".venv/bin/python -m quinn.run"
$RUN --config configs/pilot.yaml --gpus 0,1 --resume
$RUN --config configs/characterization.yaml --gpus 0,1 --resume
$RUN --config configs/characterization.yaml --gpus 0,1 --resume \
  --session isolated-reference-02 --seed 127727
```

Sweep B0, M0, M1 and M2 with unique session IDs. The pinned B0 CUDA-event schedule defaults to ten returned trials with one discarded first trial. Compare by measured cost, since equal counts can have different GPU costs.

```sh
$RUN --config configs/pilot.yaml --gpus 0,1 --resume --mode b0 --blocks 10 --session b0-10
for protocol in m0 m1 m2; do
  for blocks in 4 8 16 32 64; do
    $RUN --config configs/pilot.yaml --gpus 0,1 --resume \
      --mode "$protocol" --blocks "$blocks" --session "$protocol-$blocks"
  done
done
```

Run the smoke test and assess Gate A before scaling. The ceiling is cumulative across captures with the shared ledger. Choose the ceiling and ledger at the start of each study.

## Conditions and analysis

Use distinct sessions to compare telemetry enabled versus disabled, repeat execution across independent host sessions, and control compute or memory contention. The predefined requested host duty fractions are 25%, 50% and 75%. Measured device occupancy may differ from the requested duty. Execute the same tasks on the second device in a separate one-device run for per-task cross-device comparisons. Gold references are required separately on each device.

```sh
$RUN --config configs/pilot.yaml --gpus 0,1 --resume --session paired-telemetry-off --telemetry-every 0
$RUN --config configs/pilot.yaml --gpus 0,1 --resume --session compute-50 \
  --condition controlled-compute --interference compute --contention-duty 0.5
$RUN --config configs/pilot.yaml --gpus 0,1 --resume --session memory-50 \
  --condition controlled-memory --interference memory --contention-duty 0.5
make evaluate-measurement CAPTURES="artifacts/raw/pilot-01 artifacts/raw/isolated-reference-01 artifacts/raw/isolated-reference-02"
```

For offline scheduling, add `--replay-budgets-ms` and `--block-bound-ms` to the analysis invocation. Choose the bounds and budgets from development capture prior to held-out testing. Replay keeps within-candidate block order and uses measured block wall time as a reservation proxy. Compilation, queue time and the original global chronology fall outside that counterfactual.

For fixed service characterization, run `make evaluate-system GPUS=0` and `make evaluate-system GPUS=0,1`. Accounting summarizes task and request completions, assignment and idle time, and per-request latency for each invocation. Use `--batch-deadline-s` with a distinct session ID to check bounded admission. Online adaptive service and realistic burst workloads remain open.

Expensive RL runs require positive physical Gates A–C and restricted execution for generated code. The default reproduction workflow launches no CUDA workloads or policy training.
