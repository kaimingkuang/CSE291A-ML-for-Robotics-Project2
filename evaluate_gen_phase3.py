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
    args = parser.parse_args()
    cfg = OmegaConf.load(f"logs/{args.env}/phase1_gen.yaml")
    model_path = os.path.join(f"logs/{args.env}/phase3_gen/best_model.zip")
    res = {}

    for model_id in cfg.env.model_ids:
        print(f"Evaluating model {model_id}...")
        succ_rate = single_evaluate(cfg, str(model_id), False, model_path)
        res[model_id] = succ_rate

    with open(f"logs/{args.env}/eval_gen_phase3_res.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
