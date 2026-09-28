# Parallel shared-prefix decision benchmark

Compares sequential reusable-state decision evaluation with llama.cpp multi-sequence batching for 1, 2, 3, and 6 questions sharing the same prefix.

The benchmark pins the same Qwen3.5-4B Q4_K_M model SHA and llama.cpp revision used by the prior reproducible decision-only benchmark. It records correctness, prefix time, restore time, decode time, wall time, throughput, peak RSS, and runner metadata.

Results are tracked in GitHub Issue #1.
