import argparse
import glob
import json
import os
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate import single_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    args = parser.parse_args()
    args.seed = 42 * args.seed

    if args.by_pcd:
        spe_folders = sorted(glob.glob(f"logs/{args.env}/seed={args.seed}/phase2_spe*_by_pcd/"))
    elif args.by_orc:
        spe_folders = sorted(glob.glob(f"logs/{args.env}/seed={args.seed}/phase2_spe*_by_orc/"))
    else:
        spe_folders = sorted([x for x in glob.glob(f"logs/{args.env}/seed={args.seed}/phase2_spe*/") if ("by_pcd" not in x) and ("by_orc" not in x)])

    res = {}
    progress = tqdm(total=len(spe_folders))

    for spe_idx, spe_folder in enumerate(spe_folders):
        progress.set_description_str(f"Specialist {spe_idx}")
        if args.by_pcd:
            cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}_by_pcd.yaml")
        elif args.by_orc:
            cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}_by_orc.yaml")
        else:
            cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}.yaml")

        if args.debug:
            cfg.eval.n_final_eval_episodes = 2
            cfg.env.n_env_procs = 8

        if args.by_pcd:
            model_path = f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}_by_pcd.zip"
        elif args.by_orc:
            model_path = f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}_by_orc.zip"
        else:
            model_path = f"logs/{args.env}/seed={args.seed}/phase2_spe{spe_idx}.zip"

        res[f"spe{spe_idx}"] = {}
        avg_succ = 0
        for model_id in cfg.env.model_ids.split(","):
            succ_rate = single_evaluate(cfg, model_id, False, model_path)
            res[f"spe{spe_idx}"][model_id] = succ_rate
            avg_succ += succ_rate

        avg_succ /= len(cfg.env.model_ids.split(","))
        progress.set_description_str(f"Specialist {spe_idx}={avg_succ:.4f}")
        progress.update()

    progress.close()

    if not args.debug:
        if args.by_pcd:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase2_spe_by_pcd.json"
        elif args.by_orc:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase2_spe_by_orc.json"
        else:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase2_spe.json"
        with open(res_path, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
