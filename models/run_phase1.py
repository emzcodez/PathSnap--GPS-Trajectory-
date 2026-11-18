"""Run Prema's Phase-1 candidate generation + baseline experiment.

Usage:
  python -m models.run_phase1 --dataset "../Student folder" --train-trips 5 --test-trips 2
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import pandas as pd

from models.candidate_generation import CandidateGenerator, assign_reference_labels, candidate_miss_rate
from models.feature_extraction import build_feature_table
from models.baseline_training import train_baseline, evaluate_candidate_classifier


def load_trip(folder: Path):
    trip_id = folder.name
    gps = pd.read_csv(folder/f"{trip_id}.track", sep="\t", header=None, names=["longitude","latitude","timestamp"])
    nodes = pd.read_csv(folder/f"{trip_id}.nodes", sep="\t", header=None, names=["longitude","latitude"])
    arcs = pd.read_csv(folder/f"{trip_id}.arcs", sep="\t", header=None, names=["from_node_id","to_node_id"])
    route = pd.read_csv(folder/f"{trip_id}.route", sep="\t", header=None).iloc[:,0].to_numpy()
    return {"trip_id":trip_id,"gps":gps,"nodes":nodes,"arcs":arcs,"reference_route":route}


def prepare_trip(trip, generator):
    candidates = generator.generate(trip["gps"],trip["nodes"],trip["arcs"])
    labelled, obs_labels = assign_reference_labels(candidates,trip["gps"],trip["nodes"],trip["arcs"],trip["reference_route"])
    labelled["trip_id"] = trip["trip_id"]
    miss = candidate_miss_rate(obs_labels,candidates)
    return labelled, obs_labels, miss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True)
    parser.add_argument("--train-trips",type=int,default=5)
    parser.add_argument("--test-trips",type=int,default=2)
    parser.add_argument("--max-candidates",type=int,default=8)
    parser.add_argument("--max-distance",type=float,default=100)
    args=parser.parse_args()
    dataset=Path(args.dataset)
    train_dirs=sorted((dataset/"train").iterdir())[:args.train_trips]
    test_dirs=sorted((dataset/"student_test").iterdir())[:args.test_trips]
    generator=CandidateGenerator(max_candidates=args.max_candidates,max_distance_m=args.max_distance)
    train_frames=[]; test_frames=[]; train_misses={}; test_misses={}
    for folder in train_dirs:
        trip=load_trip(folder); frame,obs,miss=prepare_trip(trip,generator)
        train_frames.append(frame); train_misses[trip["trip_id"]]=miss
        print(f"TRAIN {trip['trip_id']}: gps={len(trip['gps'])}, candidates={len(frame)}, candidate_miss_rate={miss:.4f}")
    for folder in test_dirs:
        trip=load_trip(folder); frame,obs,miss=prepare_trip(trip,generator)
        test_frames.append(frame); test_misses[trip["trip_id"]]=miss
        print(f"STUDENT_TEST {trip['trip_id']}: gps={len(trip['gps'])}, candidates={len(frame)}, candidate_miss_rate={miss:.4f}")
    train_df=pd.concat(train_frames,ignore_index=True)
    test_df=pd.concat(test_frames,ignore_index=True)
    print(f"\nTraining candidate rows: {len(train_df)}; positive labels: {int(train_df.label.sum())}; trips: {train_df.trip_id.nunique()}")
    model,train_time=train_baseline(train_df)
    metrics,test_scored=evaluate_candidate_classifier(model,test_df)
    metrics["training_time_s"]=train_time
    metrics["train_trips"]=train_df.trip_id.nunique()
    metrics["student_test_trips"]=test_df.trip_id.nunique()
    metrics["train_candidate_miss_rate_mean"]=sum(train_misses.values())/max(len(train_misses),1)
    metrics["student_candidate_miss_rate_mean"]=sum(test_misses.values())/max(len(test_misses),1)
    out=Path(__file__).resolve().parents[1]/"results"
    out.mkdir(parents=True,exist_ok=True)
    train_df.to_csv(out/"phase1_train_candidates.csv",index=False)
    test_scored.to_csv(out/"phase1_student_test_candidates_scored.csv",index=False)
    with open(out/"phase1_baseline_metrics.json","w") as f: json.dump(metrics,f,indent=2)
    print("\nBASELINE METRICS")
    print(json.dumps(metrics,indent=2))
    print(f"\nSaved results under: {out}")

if __name__=="__main__": main()
