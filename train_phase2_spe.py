import shutil
from argparse import ArgumentParser

from omegaconf import OmegaConf

from train import train


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--spe_idx", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(f"logs/{args.env}/phase2_spe{args.spe_idx}.yaml")
    cfg.log_dir = f"logs/{args.env}/phase2_spe{args.spe_idx}"
    args.model_path = f"logs/{args.env}/phase1_gen.zip"

    if args.debug:
        cfg.train.total_steps = 10000
        cfg.eval.eval_freq = 1
        cfg.env.n_env_procs = 8
        cfg.eval.n_final_eval_episodes = 10

    train(args, cfg)
    shutil.copyfile(f"{cfg.log_dir}/best_model.zip",
        f"logs/{args.env}/phase2_spe{args.spe_idx}.zip")


if __name__ == "__main__":
    main()
