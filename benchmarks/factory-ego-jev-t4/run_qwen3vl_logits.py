#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def make_prompt(unit_queries, labels):
    lines = ["Select the action currently visible in this image."]
    for label, (_, query) in zip(labels, unit_queries.items()):
        lines.append(f"{label}: {query}")
    lines.append(f"{labels[len(unit_queries)]}: other / none of these actions")
    lines.append("Answer with exactly one label.")
    return "\n".join(lines)

def candidate_token_ids(tokenizer, labels):
    out = []
    for label in labels:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"candidate {label!r} is not one token: {ids}")
        out.append(ids[0])
    return out

def rows_to_prediction(run_id, unit_id, rows, event_ids, fps, threshold):
    events = {event_id: [] for event_id in event_ids}
    active = None
    start = None
    frame_dt = 1.0 / fps
    for row in rows:
        event_probs = [row["probs"][eid] for eid in event_ids]
        best_i = max(range(len(event_probs)), key=event_probs.__getitem__)
        best_event = event_ids[best_i]
        other_p = row["probs"]["__other__"]
        chosen = best_event if event_probs[best_i] >= threshold and event_probs[best_i] > other_p else None
        t = float(row["t"])
        if chosen != active:
            if active is not None:
                events[active].append({"start_s": start, "end_s": t})
            active, start = chosen, (t if chosen is not None else None)
    if active is not None and rows:
        events[active].append({"start_s": start, "end_s": float(rows[-1]["t"]) + frame_dt})
    return {"run_id": run_id, "unit_id": unit_id, "method": "frame_direct_logits",
            "interval_convention": "half-open_seconds", "events": events}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--unit", action="append", dest="units")
    args = ap.parse_args()

    queries = json.loads(args.queries.read_text())
    if args.units:
        wanted = set(args.units)
        queries = {k: v for k, v in queries.items() if k in wanted}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw").mkdir(exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = args.max_pixels
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa"
    ).to("cuda").eval()

    all_batch_ms = []
    all_frames = 0
    run_id = args.out.name
    for unit_id, unit_queries in queries.items():
        event_ids = list(unit_queries)
        labels = LABELS[:len(event_ids) + 1]
        token_ids = candidate_token_ids(processor.tokenizer, labels)
        prompt = make_prompt(unit_queries, labels)
        frame_dir = args.upstream / "data/factory_ego/units" / unit_id / "frames"
        frame_paths = sorted(frame_dir.glob("f*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"no frames: {frame_dir}")
        rows = []
        for batch_paths in chunks(frame_paths, args.batch_size):
            images = [Image.open(x).convert("RGB") for x in batch_paths]
            texts = []
            for image in images:
                msg = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                texts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                output = model(**inputs, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            probs = torch.softmax(output.logits[:, -1, token_ids].float(), dim=-1).cpu().tolist()
            all_batch_ms.append(dt * 1000)
            for path, pvec in zip(batch_paths, probs):
                idx = int(path.stem.lstrip("f"))
                prob_map = {eid: float(pvec[i]) for i, eid in enumerate(event_ids)}
                prob_map["__other__"] = float(pvec[len(event_ids)])
                rows.append({"idx": idx, "t": idx / args.fps, "probs": prob_map})
                all_frames += 1
            raw_path = args.out / "raw" / f"{unit_id}.json"
            raw_path.write_text(json.dumps(rows, indent=2) + "\n")
        pred = rows_to_prediction(run_id, unit_id, rows, event_ids, args.fps, args.threshold)
        (args.out / "predictions" / f"{unit_id}.json").write_text(json.dumps(pred, indent=2) + "\n")
        print(unit_id, len(rows), "frames", flush=True)

    total_s = sum(all_batch_ms) / 1000.0
    stats = {"model": args.model, "batch_size": args.batch_size, "frames": all_frames,
             "gpu_inference_seconds": total_s, "gpu_fps": all_frames / total_s if total_s else None,
             "mean_batch_ms": sum(all_batch_ms) / len(all_batch_ms) if all_batch_ms else None,
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
