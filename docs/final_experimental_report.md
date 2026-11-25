# Quinn experimental report

## Outcome

The host exposed one NVIDIA GeForce RTX 4090 Laptop GPU with 16,376 MiB of reported framebuffer memory, driver 596.08, and CUDA driver compatibility advertised as 13.2. No A100 or second device was visible, `nvcc` was absent, and permissions to change clocks remained unverified. Both the A100 pilot and hardware smoke command stopped before opening a CUDA worker. Research captures consumed 0 GPU-hours. The 96 aggregate GPU-hour ceiling remains available on a suitable host. See the [hardware probe](../artifacts/manifests/hardware.json) and [pilot preflight](../artifacts/raw/pilot-01/preflight.json).

The delivered evidence consists of CPU tests, a pinned 30-task KernelBench recipe manifest, and 864 scheduler runs over 36 generated groups. Six candidate recipes per task produce 180 planned candidates. Validity on A100 and natural rollout provenance remain pending.

## Synthetic CPU results

At the middle modeled budget, fixed pairing used mean modeled occupancy of 137.68 ms/group and yielded mean advantage RMSE of 0.279. Quinn used 137.58 modeled ms/group and yielded 0.324. The paired difference, Quinn minus fixed, was +0.0456 (exploratory 95% group-bootstrap interval +0.0014 to +0.1102). The contrast with reward-variance allocation was +0.0021 (−0.0271 to +0.0331). The contrast with rank-aware allocation was −0.0327 (−0.0671 to −0.0081). Across the three budgets, Quinn showed no consistent advantage over fixed pairing or reward-variance allocation. These are exploratory contrasts without multiplicity adjustment. See the [machine-generated summary](../artifacts/diagnostics/processed/summary.json), [cost curve](../artifacts/diagnostics/figures/fig11_advantage_cost_pareto.pdf), and [ablation table](../artifacts/diagnostics/tables/scheduler_ablation.tex).

In the correlated Gaussian perturbation check, first-order expected squared advantage error agreed closely with Monte Carlo for separated rewards (0.00105 versus 0.00105 after rounding). For the near-tie fixture it predicted 10.49 versus a Monte Carlo estimate of 3.63. At the threshold it predicted 0 versus 2.62. Exactly identical rewards also showed substantial underestimation. These are finite-noise properties of the modeled advantage and reward effects. They identify cases where local derivatives alone can misallocate samples. See the [approximation figure](../artifacts/diagnostics/figures/fig_delta_approximation.pdf) and the [raw and processed diagnostic artifacts](../artifacts/diagnostics/).

The pairing ablation exposed a benefit under generated traces where pairing was explicitly modeled to remove most common drift. The deadline ablation selected the same actions at all three budgets because the modeled deadline never bound. Do not infer deadline robustness or real drift mitigation from either result.

## Offline policy-step diagnostic

We mapped each synthetic group to a four-feature categorical softmax policy and applied one fixed-rate score-function update from uniform probabilities. The reference update uses the same feature vectors and latent rewards. At the middle budget, gradient cosine averaged 0.866 for fixed pairing (exploratory 95% interval 0.763 to 0.948) and 0.830 for Quinn (0.702 to 0.934). The feature map and reward traces are synthetic. The intervals overlap and the study has no language-model checkpoints or actual rollout gradients. See the [offline summary](../artifacts/diagnostics/processed/offline_policy_summary.json) and [per-group results](../artifacts/diagnostics/processed/offline_policy_update.csv).

## Hypotheses and gates

| Hypothesis | Status | Evidence and gap |
| --- | --- | --- |
| H1. Performance reward uncertainty | Open | Hardware admission blocked physical runtime characterization. |
| H2. Advantage distortion | Mechanism verified on synthetic inputs | Analytical and numerical derivatives agree locally. Near ties and reward discontinuities expose finite-noise failure. Natural group evidence remains open. |
| H3. Temporal pairing | Open on hardware | CPU ablation sensitivity follows the designed drift model. Paired-versus-fixed A100 data remain pending. |
| H4. Training-aware allocation | Unsupported in the CPU diagnostic | Quinn failed to consistently outperform fixed and reward-variance allocation. Physical replay and online comparison remain pending. |
| H5. Service efficiency | Open | The fixed capture coordinator has accounting and deadline admission. Real adaptive throughput, tail latency and occupancy remain unmeasured. |
| H6. Learning utility | Gated | A synthetic categorical policy-step diagnostic ran. Natural rollout data and RL training remain pending. |

Gate A is unevaluable without representative A100 measurements and independent reference sessions. Gate B is not satisfied by the synthetic comparison. Gate C awaits measured full-system performance, and Gate D awaits gradient or learning evidence. The paper can describe the advantage-risk objective, its derivation, and the identified finite-noise failure cases as preliminary engineering findings. Claims about GPU savings, architecture scale, novelty over the wider adaptive-design literature, and learning improvements require additional evidence.

## Reproduction and outstanding work

`make setup && make reproduce` verifies the CPU implementation, fetches pinned source, checks corpus provenance, and reanalyzes recorded synthetic inputs. On the A100 host, `make setup setup-gpu hardware-check corpus` followed by `make smoke-test GPUS=0` and then `make pilot GPUS=0,1` starts the measured path. The [experimental protocol](experimental_protocol.md) provides independent reference and baseline commands. The [limitations](limitations.md) cover unverified capture behavior, natural groups, online adaptive workers, and restricted generated-code execution.
