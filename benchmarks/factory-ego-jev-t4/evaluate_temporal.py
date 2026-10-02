#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.upstream / "src"))
    from small_vlm_sop_check.core.temporal import evaluate_temporal, load_annotation, load_prediction
    from small_vlm_sop_check.evaluation.compare import _aggregate
    queries = json.loads(args.queries.read_text())
    results = []
    for unit_id, unit_queries in queries.items():
        annp = args.upstream / "datasets/factory_ego/annotations/human" / f"{unit_id}.json"
        predp = args.predictions / f"{unit_id}.json"
        ann, pred = load_annotation(json.loads(annp.read_text())), load_prediction(json.loads(predp.read_text()))
        for event_id in unit_queries:
            results.append(evaluate_temporal({event_id: ann[event_id]}, {event_id: pred[event_id]}))
    overall = _aggregate(results)
    out = {"mean_tiou": overall["mean_tiou"], "tiou_at_0_5": overall["thresholds"]["tiou@0.5"],
           "gt_occurrences": overall["gt_occurrences"], "predicted_occurrences": overall["predicted_occurrences"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
