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
    args = parser.parse_args()

    phase1_res = read_json(f"logs/{args.env}/eval_gen_phase1_res.json")
    phase2_res = read_json(f"logs/{args.env}/eval_spe_phase2_res.json")
    phase3_res = read_json(f"logs/{args.env}/eval_gen_phase3_res.json")
    phase2_res_flat = {}
    for spe_id, v in phase2_res.items():
        phase2_res_flat.update(v)
    low_model_ids = list(phase2_res_flat.keys())
    phase1_all_succ = sum(list(phase1_res.values())) / len(phase1_res)
    phase1_low_succ = sum([succ for model_id, succ in phase1_res.items() if model_id in low_model_ids]) / len(phase2_res_flat)
    phase2_low_succ = sum(list(phase2_res_flat.values())) / len(phase2_res_flat)
    phase3_all_succ = sum(list(phase3_res.values())) / len(phase3_res)
    phase3_low_succ = sum([succ for model_id, succ in phase3_res.items() if model_id in low_model_ids]) / len(phase2_res_flat)
    res = [
        {"phase": 1, "subset": "all", "succ": phase1_all_succ},
        {"phase": 1, "subset": "low", "succ": phase1_low_succ},
        {"phase": 2, "subset": "low", "succ": phase2_low_succ},
        {"phase": 3, "subset": "all", "succ": phase3_all_succ},
        {"phase": 3, "subset": "low", "succ": phase3_low_succ},
    ]
    res = pd.DataFrame(res)
    print(res)


if __name__ == "__main__":
    main()
