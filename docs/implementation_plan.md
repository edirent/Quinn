# Execution plan

1. Pin upstream sources and public dataset metadata. Record local hardware and run the A100 preflight before launch.
2. Build the diagnostic corpus and correctness, timing, accounting and analysis path. Run bounded capture where target hardware is available. Evaluate Gate A using independent sessions and representative natural groups.
3. Validate reward transformations and the advantage model on CPU. Compare non-oracular policies under controlled traces, then repeat online on A100 with matched throughput and occupancy budgets. Evaluate Gate B.
4. Evaluate the full service on one and two A100s and held-out tasks before Gate C. Establish a restricted execution environment and evaluate natural groups before considering gated online training.
5. Regenerate results from raw evidence and record open, failed or satisfied gates.
