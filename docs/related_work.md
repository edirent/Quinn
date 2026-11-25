# Related work and reuse

## KernelBench

The inspected default-branch revision is `423217d9fda91e0c2d67e4a43bf62f96f6d104f1` (2026-03-05). The remote has no advertised version tags. Its package identifies itself as 0.2.0.dev0 and declares Python 3.10 and Torch ≥2.9. It provides Level 1 and Level 2 PyTorch models and input factories, randomized correctness trials, dtype-specific comparison tolerances, multiple timing backends, and adversarial checks. Quinn uses the pinned task sources and upstream FP32 tolerances. B0 calls the pinned CUDA-event timer directly, including its cold-cache protocol. The paired measurement engine uses its own schedule.

- [KernelBench source at the inspected revision](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1)

## KernelGYM and Dr. Kernel

The inspected KernelGYM revision is `3a84417f8c0efaadb215ef638b37d12e71ed20f3`. GPU workers route requests to monitored subprocess pools and use task executors for correctness and performance. The trainer extracts model code and connects to the external reward service. The evaluation toolkit implements timing and mutation/output checks. The reward manager may rerun anomalous speedups. The paper studies decoy kernels, profiling-based reward checks, reward shaping, and several multi-turn advantage methods. Multi-turn GRPO in the inspected trainer uses mean centering in its per-turn groups, while the separate single-turn outcome advantage path uses sample standard deviation, epsilon, and a small-dispersion fallback. Quinn indexes its normalized objective to the latter for terminal-only rewards, with explicit alternatives for RLOO and mean centering. A trainer integration must verify the selected advantage path and masks.

The public RL and validation artifacts are query pools, pinned to `f715c601f21e08707608a470658f5c98c51ab168` and `b0f99c544a5af05a04ae387681b79b05287fc595` respectively. The cold-start artifact `cba0ef06a5b1e3c307b7acfa8b6acb7a46578105` contains multi-turn SFT trajectories and does not establish unfiltered sibling group provenance. The collection lists 8B and 14B checkpoints, above the proposed 0.5B–1.5B initial training range. Its extraction, profiling checks and rollout integration are reuse candidates when a restricted GPU environment and a small model pipeline are available.

- [Pinned KernelGYM code](https://github.com/hkust-nlp/KernelGYM/tree/3a84417f8c0efaadb215ef638b37d12e71ed20f3)
- [Dr. Kernel paper](https://arxiv.org/abs/2602.05885)
- [Public Dr. Kernel artifacts](https://huggingface.co/collections/hkust-nlp/drkernel)

## DistRS

DistRS disaggregates reward serving and exploits request-in, batch-out semantics with batch-level priority, history-based elastic scaling and timeout-aware resource adjustment. This informs Quinn's budget/deadline interface. The publication is a conceptual comparison; a faithful implementation was not established in this inspection. DistRS targets request and resource scheduling. Quinn asks whether the marginal learning-signal benefit of an additional physical observation can guide how much measurement to perform within a request.

- [DistRS at NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/zhu-ruidong)

## Statistical and RL overlap

Pairing, randomized blocks, repeated independent execution sessions and variance propagation are established techniques. Best-arm identification and adaptive allocation already trade sampling resources for decision accuracy. Rank-aware allocation is therefore an important comparator. Sequential confidence sequences address repeated inspection through time-uniform coverage. Quinn's current working intervals have no such guarantee under dependent GPU observations. GRPO already makes rewards relative to the group, and RLOO already provides a leave-one-out baseline. The proposed contribution to test is an operational allocation objective based on expected group advantage error per GPU reservation time, tested against reward-variance and ranking objectives on the same programs and execution conditions. Novelty and learning utility remain open.

- [Kalibera and Jones, Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/)
- [Garivier and Kaufmann, Optimal Best Arm Identification with Fixed Confidence](https://proceedings.mlr.press/v49/garivier16a.html)
- [Howard et al., Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [Ahmadian et al., Back to Basics](https://arxiv.org/abs/2402.14740)
