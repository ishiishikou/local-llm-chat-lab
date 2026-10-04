#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

from heartbeat import Heartbeat


def bridge(mask, max_gap):
    x=list(mask)
    if max_gap<=0: return x
    n=len(x); i=0
    while i<n:
        if x[i]:
            i+=1; continue
        s=i
        while i<n and not x[i]: i+=1
        e=i
        if s>0 and e<n and (e-s)<=max_gap and x[s-1] and x[e]:
            for j in range(s,e): x[j]=True
    return x


def drop_short(mask,min_run):
    x=list(mask); n=len(x); i=0
    while i<n:
        if not x[i]:
            i+=1; continue
        s=i
        while i<n and x[i]: i+=1
        if i-s<min_run:
            for j in range(s,i): x[j]=False
    return x


def spans(mask,fps=2.0):
    out=[]; s=None
    for i,on in enumerate(list(mask)+[False]):
        if on and s is None: s=i
        elif not on and s is not None:
            out.append({"start_s":s/fps,"end_s":i/fps}); s=None
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--upstream",type=Path,required=True)
    ap.add_argument("--queries",type=Path,required=True)
    ap.add_argument("--raw-root",type=Path,required=True)
    ap.add_argument("--mode",choices=["choice","noul"],required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--heartbeat",type=Path,default=Path(__file__).resolve().parent/"results"/"heartbeat.json")
    args=ap.parse_args()
    heartbeat=Heartbeat(args.heartbeat,"onejev-factory").start()
    heartbeat.update(status="running",phase="postprocess-sweep",progress_current=0,progress_total=144)
    sys.path.insert(0,str(args.upstream/"src"))
    from small_vlm_sop_check.core.temporal import evaluate_temporal, load_annotation, load_prediction
    from small_vlm_sop_check.evaluation.compare import _aggregate

    queries=json.loads(args.queries.read_text())
    raw={u:json.loads((args.raw_root/f"{u}.json").read_text()) for u in queries}
    anns={}
    for u in queries:
        p=args.upstream/"datasets/factory_ego/annotations/human"/f"{u}.json"
        anns[u]=load_annotation(json.loads(p.read_text()))

    thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99]
    rows=[]
    sweep_index=0
    for th in thresholds:
      for gap in [0,1,2]:
       for min_run in [1,2,3,4]:
        results=[]
        pred_occ=0
        for u,uq in queries.items():
            rr=raw[u]
            events={}
            if args.mode=="choice":
                base={eid:[] for eid in uq}
                for eid in uq:
                    m=[]
                    for row in rr:
                        p=row["probs"]
                        top=max(p,key=p.get)
                        m.append(top==eid and p[eid]>=th)
                    m=drop_short(bridge(m,gap),min_run)
                    events[eid]=spans(m)
            else:
                for eid in uq:
                    m=[row["probs"][eid]>=th for row in rr]
                    m=drop_short(bridge(m,gap),min_run)
                    events[eid]=spans(m)
            pred=load_prediction({"unit_id":u,"run_id":"sweep","method":"onejev_sweep","interval_convention":"half-open_seconds","events":events})
            for eid in uq:
                results.append(evaluate_temporal({eid:anns[u][eid]},{eid:pred[eid]}))
        o=_aggregate(results)
        met=o["thresholds"]["tiou@0.5"]
        sweep_index+=1
        heartbeat.update(status="running",phase="postprocess-sweep",progress_current=sweep_index,progress_total=144)
        rows.append({
          "threshold":th,"gap":gap,"min_run":min_run,
          "mean_tiou":o["mean_tiou"],
          "precision":met["precision"],"recall":met["recall"],"f1":met["f1"],
          "predicted_occurrences":o["predicted_occurrences"],
        })
    rows.sort(key=lambda x: ((x["f1"] if x["f1"] is not None else -1),x["mean_tiou"]),reverse=True)
    args.out.write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps(rows[:15],indent=2))
    heartbeat.finish(status="completed",phase="postprocess-sweep:complete",progress_current=144,progress_total=144)

if __name__=="__main__": main()
