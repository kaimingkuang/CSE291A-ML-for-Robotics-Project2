import argparse
import glob
import json
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf
from tensorboard.backend.event_processing import event_accumulator

from evaluate import single_evaluate
from utils import find_best_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    args = parser.parse_args()
    args.seed = 42 * args.seed
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen.yaml")
    if args.debug:
        cfg.eval.n_final_eval_episodes = 2
        cfg.env.n_env_procs = 8

    if args.by_pcd:
        tb_path = sorted(glob.glob(f"logs/{args.env}/seed={args.seed}/phase3_gen_by_pcd/GAIL_*/events*"),
            key=lambda x: int(x.split("/")[-2].split("_")[-1]))[-1]
    elif args.by_orc:
        tb_path = sorted(glob.glob(f"logs/{args.env}/seed={args.seed}/phase3_gen_by_orc/GAIL_*/events*"),
            key=lambda x: int(x.split("/")[-2].split("_")[-1]))[-1]
    else:
        tb_path = sorted(glob.glob(f"logs/{args.env}/seed={args.seed}/phase3_gen/GAIL_*/events*"),
            key=lambda x: int(x.split("/")[-2].split("_")[-1]))[-1]
    tb = event_accumulator.EventAccumulator(tb_path)
    tb.Reload()

    if args.by_pcd:
        model_path = find_best_weight(f"logs/{args.env}/seed={args.seed}/phase3_gen_by_pcd", tb)
    elif args.by_orc:
        model_path = find_best_weight(f"logs/{args.env}/seed={args.seed}/phase3_gen_by_orc", tb)
    else:
        model_path = find_best_weight(f"logs/{args.env}/seed={args.seed}/phase3_gen", tb)
    res = {}

    for model_id in cfg.env.model_ids.split(","):
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, model_path)
        res[model_id] = succ_rate

    if not args.debug:
        if args.by_pcd:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase3_gen_by_pcd.json"
        elif args.by_orc:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase3_gen_by_orc.json"
        else:
            res_path = f"logs/{args.env}/seed={args.seed}/eval_phase3_gen.json"
        with open(res_path, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
