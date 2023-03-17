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
    args = parser.parse_args()

    spe_folders = sorted(glob.glob(f"logs/{args.env}/phase2_spe*"))

    res = {}
    progress = tqdm(total=len(spe_folders))

    for spe_idx, spe_folder in enumerate(spe_folders):
        progress.set_description_str(f"Specialist {spe_idx}")
        cfg = OmegaConf.load(f"logs/{args.env}/phase2_spe{spe_idx}.yaml")
        model_path = os.path.join(spe_folder, "best_model.zip")
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

    with open(f"logs/{args.env}/eval_phase2_spe.json", "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
