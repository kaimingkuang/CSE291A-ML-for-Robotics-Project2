import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf

from evaluate import single_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.seed = 42 * args.seed
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen/phase1_gen.yaml")
    if args.debug:
        cfg.eval.n_final_eval_episodes = 2
        cfg.env.n_env_procs = 8
    model_path = f"logs/{args.env}/seed={args.seed}/phase1_gen.zip"
    res = {}

    for model_id in cfg.env.model_ids.split(","):
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, model_path)
        res[model_id] = succ_rate

    save_dir = f"logs/{args.env}/seed={args.seed}"
    if not args.debug:
        with open(os.path.join(save_dir, "eval_phase1_gen.json"), "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
