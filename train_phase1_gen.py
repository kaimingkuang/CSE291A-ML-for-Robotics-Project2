import os
import shutil
from argparse import ArgumentParser

from omegaconf import OmegaConf

from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.model_path = None
    args.seed = 42 * args.seed

    log_dir = f"logs/{args.env}/seed={args.seed}/phase1_gen"
    os.makedirs(log_dir, exist_ok=True)

    shutil.copyfile(f"configs/{args.env}_phase1_gen_base.yaml",
        f"{log_dir}/phase1_gen.yaml")
    cfg = OmegaConf.load(f"{log_dir}/phase1_gen.yaml")
    cfg.trial_name = f"{args.env}_seed={args.seed}_phase1_gen"
    cfg.log_dir = log_dir

    if args.debug:
        cfg.train.total_steps = 10000
        cfg.eval.eval_freq = 1
        cfg.env.n_env_procs = 8
        cfg.eval.n_final_eval_episodes = 10

    train(args, cfg)


if __name__ == "__main__":
    main()
