import json
import os
from argparse import ArgumentParser

import pandas as pd


def read_json(json_path):
    with open(json_path, "r") as f:
        content = json.load(f)

    return content


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    args = parser.parse_args()
    args.seed = 42 * args.seed

    phase1_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase1_gen.json")

    if args.by_pcd:
        phase2_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase2_spe_by_pcd.json")
        phase3_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase3_gen_by_pcd.json")
    elif args.by_orc:
        phase2_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase2_spe_by_orc.json")
        phase3_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase3_gen_by_orc.json")
    else:
        phase2_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase2_spe.json")
        phase3_res = read_json(f"logs/{args.env}/seed={args.seed}/eval_phase3_gen.json")
    phase2_res_flat = {}
    for spe_id, v in phase2_res.items():
        phase2_res_flat.update(v)
    low_model_ids = list(phase2_res_flat.keys())

    phase1_res = pd.DataFrame([{"group": "high", "model_id": model_id, "phase1_succ": succ} for model_id, succ in phase1_res.items()])
    phase1_res.loc[phase1_res.model_id.isin(low_model_ids), "group"] = "low"
    phase3_res = pd.DataFrame([{"group": "high", "model_id": model_id, "phase3_succ": succ} for model_id, succ in phase3_res.items()])
    phase3_res.loc[phase3_res.model_id.isin(low_model_ids), "group"] = "low"
    res = phase1_res.merge(phase3_res, on=["group", "model_id"], how="left")
    res["diff"] = res["phase3_succ"] - res["phase1_succ"]
    print(res.groupby("group")["diff"].mean())


if __name__ == "__main__":
    main()
