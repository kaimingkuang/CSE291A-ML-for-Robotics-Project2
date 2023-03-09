import random

import numpy as np
import argparse
import json
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", required=True,
        help="Phase I generalist evaluation results.")
    parser.add_argument("--succ-thresh", default=0.7,
        help="The success rate upper bound requiring specialist training.")
    parser.add_argument("--n-model-spe", default=5,
        help="The number of models each specialist is trained on.")
    parser.add_argument("--n-steps-spe", default=1500000, type=int,
        help="The number of training steps for each specialist.")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--spe-init", required=True,
        help="The model path to initialize each specialist.")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    
    with open(args.eval_json, "r") as f:
        eval_res = json.load(f)
    
    low_perf_models = []
    high_perf_models = []
    low_perf_succ = 0
    high_perf_succ = 0
    overall_perf_succ = 0
    for model_id, succ_rate in eval_res.items():
        overall_perf_succ += succ_rate
        if succ_rate < args.succ_thresh:
            low_perf_models.append(model_id)
            low_perf_succ += succ_rate
        else:
            high_perf_models.append(model_id)
            high_perf_succ += succ_rate
    
    low_perf_succ /= len(low_perf_models)
    high_perf_succ /= len(high_perf_models)
    overall_perf_succ /= (len(low_perf_models) + len(high_perf_models))

    n_high_models = len(high_perf_models)
    n_low_models = len(low_perf_models)
    n_models = (len(low_perf_models) + len(high_perf_models))
    print(f"The generalist scores >={args.succ_thresh} success rate on {n_high_models}/{n_models} models: {high_perf_models}.")
    print(f"Average success rates on these models: {high_perf_succ:.4f}.\n")
    print(f"The generalist scores <{args.succ_thresh} success rate on {n_low_models}/{n_models} models: {low_perf_models}.")
    print(f"Average success rates on these models: {low_perf_succ:.4f}.\n")
    print(f"Overall average success rates: {overall_perf_succ:.4f}.")

    n_spes = n_low_models // args.n_model_spe
    random.shuffle(low_perf_models)
    grouping = {}
    for i in range(n_spes):
        if i != n_spes - 1:
            beg = i * args.n_model_spe
            end = (i + 1) * args.n_model_spe
        else:
            beg = i * args.n_model_spe
            end = n_low_models
        grouping[f"spe{i}"] = low_perf_models[beg:end]
    
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    trial_name = cfg.trial_name.replace("gen_phase1", "spe_phase2")
    for spe_id, model_ids in grouping.items():
        cfg.env.model_ids = ",".join(model_ids)
        cfg.trial_name = f"{trial_name}_{spe_id}"
        cfg.train.total_steps = args.n_steps_spe
        cfg.init_model_path = args.spe_init
        with open(f"configs/{args.cfg.replace('gen_phase1', 'spe_phase2')}_{spe_id}.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)


if __name__ == "__main__":
    main()
