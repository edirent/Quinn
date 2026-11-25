# Experiment entry points and status

| Experiment | Entry point | Available evidence |
| --- | --- | --- |
| 1. Characterization | `make pilot`, `make characterize`, `make evaluate-measurement CAPTURES="…"` | Hardware admission blocked |
| 2. Advantage distortion | `make evaluate-scheduler` and A100 capture analysis | Synthetic approximation diagnostic |
| 3. Measurement engine | M0/M1/M2 and B0 capture, followed by `make evaluate-measurement` | Timing interface tests |
| 4. Scheduler | `make evaluate-scheduler`; physical replay via `--analyze-capture` | Synthetic CPU comparisons |
| 5. Reward service | `make evaluate-system` | Fixed capture coordinator, A100 admission blocked |
| 6. Ablation | `make ablation` | Synthetic table and per-group outcomes |
| 7. Generalization | `make generalization` | Held-out manifest |
| 8. Online RL | `make train-rl`; `make report` for offline diagnostic | Online gate closed. Synthetic categorical policy-step data available. |
