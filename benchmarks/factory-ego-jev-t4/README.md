# Factory Ego Jev-style T4 benchmark

Purpose: compare Jev-style direct-logit action classification with the existing
`shure-dev/small-vlm-sop-check` Factory Ego benchmark on a Colab Tesla T4.

## Rules
- Upstream benchmark is pinned to commit `eb52106d5684a4e951ea4cf2430818e673d228dc`.
- Inference uses the existing run `queries.json`; human GT text/intervals are not
  used as model input.
- No prose generation: one forward pass reads candidate label logits.
- Every frame is new. No same-frame vision-embedding cache/reuse.
- TensorRT is intentionally excluded.
- Primary sampling rate is 2 fps.
- Report accuracy and throughput separately, then derive camera capacity from
  p95 latency/deadline and sustained throughput.

## Data
Factory Ego source media comes from gated `builddotai/Egocentric-10K`.
Media is never committed. Use `fetch_factory_ego.py` to reconstruct the 20
Factory Ego units under the pinned upstream repo.

## Outputs
`results/` stores small JSON/Markdown benchmark summaries only.
Large media/model caches stay outside Git.

## First target
Qwen3-VL-2B-Instruct direct logits, then Qwen3-VL-4B and MiniCPM-V variants.
