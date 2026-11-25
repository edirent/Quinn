# Limitations

- Zero physical A100s were accessible. CUDA launches, interference, compilation timing, GPU watchdog behavior, across-device agreement and actual occupancy are unverified. The local RTX 4090 Laptop device was identified during hardware preflight.
- The corpus provides 180 planned recipes. The count of functionally correct candidates remains unknown. Many recipes are compiler or execution variants of the same implementation. Three unary tasks use maintained Triton variants. Natural rollout groups, the expanded 80–100-task corpus, and generated CUDA support remain pending.
- The worker limits environment inheritance and runtime but executes trusted code as the current user. Generated programs require namespace, network, and filesystem isolation before rollout capture or training.
- Captures assume operator-declared isolation and log supported clock data. The host's clock-control permissions and process isolation were unverified. Dedicated cold-start and causal carryover analyses remain pending.
- B0 reuses the KernelBench timer. The KernelGYM transform implements its configurable clipped-speedup component with correctness bonus and a fixed invalid penalty. Coverage-based and multi-turn shaping and the full trainer's reward-hacking checks remain separate mechanisms.
- Replay preserves ordered per-candidate streams and needs declared admission bounds. Simulated occupancy and block-wall proxies are insufficient for two-worker adaptive cost claims. Physical replay currently requires complete all-correct candidate groups. CPU simulations include mixed validity.
- CPU bootstrap intervals are exploratory. Calibration on adaptive GPU measurements and policy-gradient or training effects remain open.
