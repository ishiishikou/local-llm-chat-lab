#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from heartbeat import Heartbeat

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def prompt_for(queries):
    labels = LABELS[:len(queries) + 1]
    lines = ["Select the action currently visible in this image."]
    for label, query in zip(labels, queries.values()):
        lines.append(f"{label}: {query}")
    lines.append(f"{labels[len(queries)]}: other / none of these actions")
    lines.append("Answer with exactly one label.")
    return "\n".join(lines)

def percentile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--min-pixels", type=int, default=3136)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heartbeat", type=Path, default=Path(__file__).resolve().parent / "results" / "heartbeat.json")
    args = ap.parse_args()

    heartbeat = Heartbeat(args.heartbeat, "onejev-factory").start()
    heartbeat.update(status="running", phase="throughput:model-load", model=args.model)
    queries = json.loads(args.queries.read_text())
    items = []
    for unit_id, qs in queries.items():
        prompt = prompt_for(qs)
        frame_dir = args.upstream / "data/factory_ego/units" / unit_id / "frames"
        for frame in sorted(frame_dir.glob("f*.jpg")):
            items.append((frame, prompt))
    if args.limit:
        items = items[:args.limit]
    print("ITEMS", len(items), flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size["longest_edge"] = args.max_pixels
    processor.image_processor.size["shortest_edge"] = args.min_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda").eval()

    results = []
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        heartbeat.update(status="running", phase=f"throughput:batch-{bs}", progress_current=0, progress_total=len(items))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_ms, e2e_ms = [], []
        n = 0

        warm = items[:min(bs, len(items))]
        images = [Image.open(f).convert("RGB") for f, _ in warm]
        texts = []
        for image, (_, prompt) in zip(images, warm):
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            texts.append(processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ))

        inp = processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )
        inp = {k: (v.to("cuda") if hasattr(v, "to") else v)
               for k, v in inp.items()}
        with torch.inference_mode():
            _ = model(**inp, use_cache=False, logits_to_keep=1)
        torch.cuda.synchronize()

        wall0 = time.perf_counter()
        for start in range(0, len(items), bs):
            batch = items[start:start + bs]
            t0 = time.perf_counter()
            images = [Image.open(f).convert("RGB") for f, _ in batch]
            texts = []
            for image, (_, prompt) in zip(images, batch):
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                texts.append(processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                ))
            inp = processor(
                text=texts, images=images, padding=True, return_tensors="pt"
            )

            inp = {k: (v.to("cuda") if hasattr(v, "to") else v)
                   for k, v in inp.items()}
            torch.cuda.synchronize()
            g0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(**inp, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            g1 = time.perf_counter()
            gpu_ms.append((g1 - g0) * 1000)
            e2e_ms.append((g1 - t0) * 1000)
            n += len(batch)
            heartbeat.update(status="running", phase=f"throughput:batch-{bs}", progress_current=n, progress_total=len(items))

        wall = time.perf_counter() - wall0
        rec = {
            "batch_size": bs,
            "frames": n,
            "wall_fps": n / wall,
            "gpu_fps": n / (sum(gpu_ms) / 1000),
            "batch_e2e_p50_ms": percentile(e2e_ms, 0.50),
            "batch_e2e_p95_ms": percentile(e2e_ms, 0.95),
            "batch_gpu_p50_ms": percentile(gpu_ms, 0.50),
            "batch_gpu_p95_ms": percentile(gpu_ms, 0.95),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "capacity_2fps_raw": math.floor((n / wall) / 2),
        }

        results.append(rec)
        print("RESULT", json.dumps(rec, sort_keys=True), flush=True)

    out = {
        "model": args.model,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels,
        "frame_count": len(items),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("WROTE", args.out, flush=True)
    heartbeat.finish(status="completed", phase="throughput:complete")

if __name__ == "__main__":
    main()
