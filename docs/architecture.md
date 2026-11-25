# Architecture

`quinn.run` accepts YAML configurations and command-line overrides. It dispatches to a physical capture coordinator, the synthetic trace experiment, or artifact analysis.

- The workload adapter reads pinned KernelBench Git objects. The manifest enumerates trusted diagnostic variants, their hashes, task splits and operator categories.
- The coordinator leases up to two physical A100s and assigns whole tasks to devices. Each candidate runs in a bounded subprocess with a local cache, a limited inherited environment, and a process-group watchdog. Only the pinned reference and maintained candidate recipes enter this path.
- The worker records compilation, randomized correctness checks, timing events and out-of-band telemetry. Physical accounting measures assignment wall time through process teardown. CUDA event time is available independently of this reservation charge.
- The transactional GPU-time ledger reserves the full worker bound up front. On successful teardown it charges the measured assignment. A coordinator interruption with ambiguous completion retains the reservation. A fully closed raw capture can be adopted after an interruption.
- The scheduler consumes observations through a block source with a cost and deadline quote. Fixed, reward-variance, rank and Quinn policies share the same admission loop. This scheduler currently runs in the CPU study and offline capture analysis. The physical coordinator uses fixed capture protocols.
- Reports read raw evidence and use separate paths for A100 capture and synthetic CPU analyses. Source data digests are recorded with processed outputs.
