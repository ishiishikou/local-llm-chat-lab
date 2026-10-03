# Factory Ego Jev-style T4 results

Dataset: Factory Ego development set, 20 units, 68 events, 81 GT occurrences, 20 s/unit at 2 fps (800 frames total).

## Data reproducibility
The source repo is pinned to `eb52106d5684a4e951ea4cf2430818e673d228dc`.
Frames were extracted at the same requested timestamps as upstream Factory Ego, but JPEG byte hashes do not match its historical manifest in the current runtime. `data/factory-ego-fixed.lock.json` freezes the exact 800 frame bytes used for all runs below. Therefore comparisons with historical upstream runs are comparable at source clip/timestamp/query/GT level, but not byte-for-byte image identity.

## Qwen3-VL-2B direct-logit throughput on Tesla T4
Model: `Qwen/Qwen3-VL-2B-Instruct`, FP16, SDPA, max_pixels=50176, no generation, no same-frame embedding reuse.

| Batch | Wall fps | GPU fps | E2E p95 | Peak VRAM | Raw cameras @2fps |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.39 | 11.99 | 140 ms | 4.29 GB | 4 |
| 2 | 14.08 | 19.97 | 177 ms | 4.32 GB | 7 |
| 4 | 15.52 | 22.46 | 324 ms | 4.37 GB | 7 |
| 8 | 16.06 | 23.11 | 606 ms | 4.47 GB | 8 |
| 16 | 16.48 | 23.92 | 1132 ms | 4.68 GB | 8 |

For a 2 fps stream, batch 4 is the largest tested batch whose p95 end-to-end batch latency stays below the 500 ms frame period. It sustains 15.52 fps, equivalent to 7 raw 2-fps camera streams. A 75% operational headroom target is about 5 cameras/T4.

## Accuracy
Historical upstream reference:
- Marlin-2B: mean tIoU 0.399861, tIoU@0.5 F1 0.510067.
- Qwen3-VL-2B autoregressive baseline: mean tIoU 0.110431, F1 0.098361.

Jev-style Qwen3-VL-2B:
- Categorical direct logits, threshold 0.5: mean tIoU 0.296892, F1 0.160279.
- Binary per-event direct logits, threshold 0.5: mean tIoU 0.409984, F1 0.156770, but it over-predicts 340 occurrences.
- Binary temporal post-process sweep best F1: mean tIoU 0.342487, F1 0.306122 (threshold 0.5, gap 2, min_run 2).
- Categorical label-prior calibration/post-process exploratory best: mean tIoU 0.346642, F1 0.346369.

The calibration/post-process sweeps were selected on the same development set, so their best scores are exploratory upper bounds rather than an unbiased generalization estimate.

## Capacity conclusion
The efficient categorical direct-logit path is the serving candidate. On this T4, use batch 4 as the measured 2-fps operating point: 7 cameras raw, approximately 5 cameras with 25% throughput headroom. The current Qwen3-VL-2B method does not yet match Marlin-2B's F1, so the original goal of Marlin-level accuracy and maximum stream density is not achieved by this model/configuration alone.
