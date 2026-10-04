#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, statistics, time
from pathlib import Path

import torch
from PIL import Image

from qev.calibrate import Calibration
from qev.mm_engine import MMDecisionEngine, local_model_dir
from qev.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest

from heartbeat import Heartbeat


def percentile(xs, q):
    if not xs:
        return None
    ys=sorted(xs)
    p=(len(ys)-1)*q
    lo,hi=math.floor(p),math.ceil(p)
    if lo==hi:
        return ys[lo]
    return ys[lo]+(ys[hi]-ys[lo])*(p-lo)


def mask_to_spans(mask, fps):
    spans=[]
    start=None
    for i,on in enumerate(mask+[False]):
        if on and start is None:
            start=i
        elif not on and start is not None:
            spans.append({"start_s":start/fps,"end_s":i/fps})
            start=None
    return spans


def choice_request(unit_queries):
    criteria={eid:q for eid,q in unit_queries.items()}
    criteria["__other__"]="none of these actions is currently being performed"
    return SystemOneRequest(
        state={"camera":"<image:1>"},
        questions={"action":ChoiceQuestion(
            type="choice",
            instructions="Which action is the worker currently performing in this image?",
            criteria=criteria,
        )}
    )


def noul_request(unit_queries):
    qs={}
    for eid,q in unit_queries.items():
        qs[eid]=NoulQuestion(
            type="noul",
            instructions=f"Is the worker currently performing this action? {q}",
        )
    return SystemOneRequest(state={"camera":"<image:1>"},questions=qs)


def one(engine, req, image_path):
    t0=time.perf_counter()
    img=Image.open(image_path).convert("RGB")
    torch.cuda.synchronize()
    g0=time.perf_counter()
    resp,meta=engine.decide(req,media=[{"type":"image","image":img}],debias=1,debug=False)
    torch.cuda.synchronize()
    g1=time.perf_counter()
    return resp,meta,(g1-t0)*1000,(g1-g0)*1000


def run_mode(engine, queries, data_root, outdir, mode, fps, threshold, heartbeat=None, limit_units=0, limit_frames=0):
    lat_e2e=[]
    lat_engine=[]
    raw_dir=outdir/mode/"raw"
    pred_dir=outdir/mode/"predictions"
    raw_dir.mkdir(parents=True,exist_ok=True)
    pred_dir.mkdir(parents=True,exist_ok=True)
    total=0
    timed_wall_s=0.0
    unit_items=list(queries.items())
    if limit_units:
        unit_items=unit_items[:limit_units]
    for unit_id,unit_queries in unit_items:
        frame_paths=sorted((data_root/"units"/unit_id/"frames").glob("f*.jpg"))
        if limit_frames:
            frame_paths=frame_paths[:limit_frames]
        req=choice_request(unit_queries) if mode=="choice" else noul_request(unit_queries)
        # Warm the exact unit/question shape once; exclude graph capture and first-use allocator work.
        if frame_paths:
            one(engine,req,frame_paths[0])
        rows=[]
        unit_t0=time.perf_counter()
        for idx,p in enumerate(frame_paths):
            resp,meta,e2e_ms,engine_ms=one(engine,req,p)
            lat_e2e.append(e2e_ms)
            lat_engine.append(engine_ms)
            if mode=="choice":
                ans=resp.answers["action"]
                probs={k:float(v) for k,v in ans.probabilities.items()}
            else:
                probs={eid:float(resp.answers[eid].noul) for eid in unit_queries}
            rows.append({"idx":idx,"t":idx/fps,"probs":probs})
            total+=1
            if heartbeat is not None:
                heartbeat.update(status="running", phase=f"inference:{mode}", progress_current=total)
        timed_wall_s += time.perf_counter()-unit_t0
        (raw_dir/f"{unit_id}.json").write_text(json.dumps(rows,indent=2)+"\n")
        events={}
        if mode=="choice":
            eids=list(unit_queries)
            masks={eid:[False]*len(rows) for eid in eids}
            for i,row in enumerate(rows):
                p=row["probs"]
                chosen=max(p,key=p.get)
                if chosen!="__other__" and p[chosen]>=threshold:
                    masks[chosen][i]=True
            for eid in eids:
                events[eid]=mask_to_spans(masks[eid],fps)
        else:
            for eid in unit_queries:
                events[eid]=mask_to_spans([r["probs"][eid]>=threshold for r in rows],fps)
        pred={
            "run_id":outdir.name+"-"+mode,
            "unit_id":unit_id,
            "method":f"onejev_{mode}",
            "interval_convention":"half-open_seconds",
            "events":events,
        }
        (pred_dir/f"{unit_id}.json").write_text(json.dumps(pred,indent=2)+"\n")
        print("UNIT",mode,unit_id,len(rows),flush=True)
    wall=timed_wall_s
    stats={
        "mode":mode,
        "frames":total,
        "wall_seconds":wall,
        "wall_fps":total/wall if wall else None,
        "capacity_2fps_raw":math.floor((total/wall)/2) if wall else None,
        "request_e2e_p50_ms":percentile(lat_e2e,.5),
        "request_e2e_p95_ms":percentile(lat_e2e,.95),
        "engine_p50_ms":percentile(lat_engine,.5),
        "engine_p95_ms":percentile(lat_engine,.95),
        "mean_e2e_ms":statistics.mean(lat_e2e) if lat_e2e else None,
        "peak_vram_gb":torch.cuda.max_memory_allocated()/1e9,
    }
    (outdir/mode/"stats.json").write_text(json.dumps(stats,indent=2)+"\n")
    print("STATS",json.dumps(stats,sort_keys=True),flush=True)
    return stats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--queries",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--modes",default="choice,noul")
    ap.add_argument("--fps",type=float,default=2.0)
    ap.add_argument("--threshold",type=float,default=0.5)
    ap.add_argument("--dtype",default="float16")
    ap.add_argument("--head-dtype",default="float32")
    ap.add_argument("--max-pixels",type=int,default=50176)
    ap.add_argument("--min-pixels",type=int,default=3136)
    ap.add_argument("--fork-mode",default="auto",choices=["auto","batched","sequential"])
    ap.add_argument("--no-cuda-graphs",action="store_true")
    ap.add_argument("--limit-units",type=int,default=0)
    ap.add_argument("--limit-frames",type=int,default=0)
    ap.add_argument("--heartbeat",type=Path,default=Path(__file__).resolve().parent/"results"/"heartbeat.json")
    args=ap.parse_args()

    heartbeat=Heartbeat(args.heartbeat,"onejev-factory").start()
    heartbeat.update(status="running",phase="model-load",model=args.model)
    args.out.mkdir(parents=True,exist_ok=True)
    queries=json.loads(args.queries.read_text())
    model_dir=Path(local_model_dir(args.model))
    calib_path=model_dir/"calibration.json"
    calib=Calibration.load(str(calib_path)) if calib_path.exists() else None
    torch.cuda.reset_peak_memory_stats()
    load0=time.perf_counter()
    engine=MMDecisionEngine(
        str(model_dir),
        device="cuda:0",
        dtype=args.dtype,
        head_dtype=args.head_dtype,
        calibration=calib,
        fork_mode=args.fork_mode,
        gpu_preprocess=True,
        cuda_graphs=not args.no_cuda_graphs,
    )
    load_s=time.perf_counter()-load0
    ip=getattr(engine.processor,"image_processor",None)
    if ip is not None and hasattr(ip,"size"):
        ip.size["longest_edge"]=args.max_pixels
        ip.size["shortest_edge"]=args.min_pixels
    env={
        "model":args.model,
        "model_dir":str(model_dir),
        "qev_commit":"81ce62f1597c91e46767d4d02d6ac2e18534fe94",
        "gpu":torch.cuda.get_device_name(0),
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "dtype":args.dtype,
        "head_dtype":args.head_dtype,
        "max_pixels":args.max_pixels,
        "min_pixels":args.min_pixels,
        "fork_mode":engine.info.fork_mode,
        "fork_max_abs_diff":engine.info.fork_max_abs_diff,
        "calibration":str(calib_path) if calib_path.exists() else None,
        "load_seconds":load_s,
        "vram_after_load_gb":torch.cuda.memory_allocated()/1e9,
    }
    (args.out/"env.json").write_text(json.dumps(env,indent=2)+"\n")
    print("ENV",json.dumps(env,sort_keys=True),flush=True)
    all_stats={}
    for mode in [x.strip() for x in args.modes.split(",") if x.strip()]:
        heartbeat.update(status="running",phase=f"inference:{mode}",progress_current=0)
        all_stats[mode]=run_mode(engine,queries,args.data_root,args.out,mode,args.fps,args.threshold,heartbeat,args.limit_units,args.limit_frames)
    (args.out/"summary.json").write_text(json.dumps({"env":env,"modes":all_stats},indent=2)+"\n")
    heartbeat.finish(status="completed",phase="complete")

if __name__=="__main__":
    main()
