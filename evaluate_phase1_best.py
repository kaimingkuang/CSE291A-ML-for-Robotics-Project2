import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf

from evaluate import single_evaluate
from utils import find_best_step, read_tensorboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.seed = 42 * args.seed
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen/phase1_gen.yaml")
    if args.debug:
        cfg.eval.n_final_eval_episodes = 10
        cfg.env.n_env_procs = 8
    
    tb_path = "logs/TurnFaucet-v2/seed=42/phase1_gen/SAC_1/events.out.tfevents.1677978341.kk-turnfaucet-gen-phase1-sac-rpwb8.13.0"
    tb_log = read_tensorboard(tb_path)
    best_step = find_best_step(tb_log)
    model_path = f"logs/{args.env}/seed={args.seed}/phase1_gen/rl_model_{best_step}_steps.zip"
    res = {}

    avg_succ_rate = 0
    for model_id in cfg.env.model_ids.split(","):
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, model_path)
        res[model_id] = succ_rate
        avg_succ_rate += succ_rate
    
    avg_succ_rate /= len(cfg.env.model_ids.split(","))
    print(f"Average success rate: {avg_succ_rate:.4f}")


if __name__ == "__main__":
    main()
