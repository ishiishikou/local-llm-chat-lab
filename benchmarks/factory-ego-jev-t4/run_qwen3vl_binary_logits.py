#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def make_prompt(query):
    return (
        "Determine whether the worker is currently performing the specified action "
        "in this single image. Objects merely being present do not count.\n"
        f"Action: {query}\n"
        "Answer Yes or No only."
    )

def percentile(values, q):
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)

def mask_to_spans(mask, fps):
    spans = []
    start = None
    for i, on in enumerate(mask + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append({"start_s": start / fps, "end_s": i / fps})
            start = None
    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--min-pixels", type=int, default=3136)
    args = ap.parse_args()

    queries = json.loads(args.queries.read_text())
    items = []
    for unit_id, unit_queries in queries.items():
        frame_dir = args.upstream / "data/factory_ego/units" / unit_id / "frames"
        for frame_idx, frame in enumerate(sorted(frame_dir.glob("f*.jpg"))):
            for event_id, query in unit_queries.items():
                items.append((unit_id, frame_idx, frame, event_id, query))

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size["longest_edge"] = args.max_pixels
    processor.image_processor.size["shortest_edge"] = args.min_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda").eval()
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    probs = defaultdict(lambda: defaultdict(dict))
    gpu_ms, e2e_ms = [], []
    torch.cuda.reset_peak_memory_stats()
    wall0 = time.perf_counter()

    for batch in chunks(items, args.batch_size):
        t0 = time.perf_counter()
        images = [Image.open(x[2]).convert("RGB") for x in batch]
        texts = []
        for image, item in zip(images, batch):
            msg = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": make_prompt(item[4])},
            ]}]
            texts.append(processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ))
        inp = processor(text=texts, images=images, padding=True, return_tensors="pt")
        inp = {k: (v.to("cuda") if hasattr(v, "to") else v)
               for k, v in inp.items()}
        torch.cuda.synchronize()
        g0 = time.perf_counter()
        with torch.inference_mode():
            output = model(**inp, use_cache=False, logits_to_keep=1)
        z = output.logits[:, -1, [yes_id, no_id]].float()
        torch.cuda.synchronize()
        g1 = time.perf_counter()
        pyes = torch.softmax(z, dim=-1)[:, 0].cpu().tolist()
        gpu_ms.append((g1 - g0) * 1000)
        e2e_ms.append((g1 - t0) * 1000)
        for item, p in zip(batch, pyes):
            unit_id, frame_idx, _, event_id, _ = item
            probs[unit_id][frame_idx][event_id] = float(p)

    wall = time.perf_counter() - wall0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)

    for unit_id, unit_queries in queries.items():
        rows = []
        for idx in range(40):
            rows.append({
                "idx": idx,
                "t": idx / args.fps,
                "probs": probs[unit_id][idx],
            })
        (args.out / "raw" / f"{unit_id}.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        events = {}
        for event_id in unit_queries:
            mask = [row["probs"][event_id] >= args.threshold for row in rows]
            events[event_id] = mask_to_spans(mask, args.fps)
        pred = {
            "run_id": args.out.name,
            "unit_id": unit_id,
            "method": "frame_binary_direct_logits",
            "interval_convention": "half-open_seconds",
            "events": events,
        }
        (args.out / "predictions" / f"{unit_id}.json").write_text(
            json.dumps(pred, indent=2) + "\n"
        )

    event_count = sum(len(x) for x in queries.values())
    avg_events = event_count / len(queries)
    pair_fps = len(items) / wall
    stats = {
        "model": args.model,
        "batch_size": args.batch_size,
        "pairs": len(items),
        "frames": 40 * len(queries),
        "events": event_count,
        "avg_events_per_frame": avg_events,
        "pair_wall_fps": pair_fps,
        "frame_equiv_fps": pair_fps / avg_events,
        "capacity_2fps_raw": math.floor((pair_fps / avg_events) / 2),
        "batch_e2e_p50_ms": percentile(e2e_ms, 0.50),
        "batch_e2e_p95_ms": percentile(e2e_ms, 0.95),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)

if __name__ == "__main__":
    main()
