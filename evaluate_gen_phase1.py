import argparse
import json
import os
import warnings
warnings.filterwarnings("ignore")

from omegaconf import OmegaConf

from evaluate import single_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Config name.")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    res = {}

    for model_id in cfg.env.model_ids:
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, args.model_path)
        res[model_id] = succ_rate
    
    with open("eval_gen_phase1_res.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
