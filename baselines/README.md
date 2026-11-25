# Baseline entry points

- B0 invokes pinned KernelBench CUDA event timing through the capture worker.
- B1 and B2 are fixed conventional and stabilized capture protocols.
- B3 is fixed paired capture and fixed allocation in the CPU diagnostic.
- B4 is cost-aware predicted reward-risk reduction.
- B5 is cost-aware expected pairwise log-speedup order-error reduction.
- B6 is cost-aware predicted advantage-risk reduction.
- B7 is an independently invoked reference capture with a separate session ID.

The replay experiment compares B3–B6 against common inputs. B0–B2 require target hardware for capture and analysis. DistRS is a literature comparator.
