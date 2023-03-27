import glob
import shutil
from argparse import ArgumentParser

from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator

from train import train
from utils import find_best_weight


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--spe-idx", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    args = parser.parse_args()
    args.seed = args.seed * 42
    args.spe_idx = args.spe_idx + args.start_idx

    if args.by_pcd:
        cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_pcd.yaml")
        cfg.log_dir = f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_pcd"
        cfg.trial_name = f"{args.env}_spe{args.spe_idx}_by_pcd"
    elif args.by_orc:
        cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_orc.yaml")
        cfg.log_dir = f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_orc"
        cfg.trial_name = f"{args.env}_spe{args.spe_idx}_by_orc"
    else:
        cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}.yaml")
        cfg.log_dir = f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}"
        cfg.trial_name = f"{args.env}_spe{args.spe_idx}"
    args.model_path = f"logs/{args.env}/seed={args.seed}/phase1_gen.zip"

    if args.debug:
        cfg.train.total_steps = 10000
        cfg.eval.eval_freq = 1
        cfg.env.n_env_procs = 8
        cfg.eval.n_final_eval_episodes = 10

    train(args, cfg)

    tb_path = sorted(glob.glob(f"{cfg.log_dir}/SAC_*/events*"),
        key=lambda x: int(x.split("/")[-2].split("_")[-1]))[-1]
    tb = event_accumulator.EventAccumulator(tb_path)
    tb.Reload()
    best_ckpt = find_best_weight(cfg.log_dir, tb)

    if args.by_pcd:
        shutil.copyfile(best_ckpt, f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_pcd.zip")
    elif args.by_orc:
        shutil.copyfile(best_ckpt, f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}_by_orc.zip")
    else:
        shutil.copyfile(best_ckpt, f"logs/{args.env}/seed={args.seed}/phase2_spe{args.spe_idx}.zip")


if __name__ == "__main__":
    main()
