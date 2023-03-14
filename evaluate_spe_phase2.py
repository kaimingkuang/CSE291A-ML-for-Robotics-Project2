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
    parser.add_argument("--cfg", required=True, help="Config name.")
    args = parser.parse_args()

    spe_folders = sorted(glob.glob(os.path.join("logs", f"{args.cfg}_spe*")))

    res = {}
    progress = tqdm(total=len(spe_folders))

    for spe_idx, spe_folder in enumerate(spe_folders):
        progress.set_description_str(f"Specialist {spe_idx}")
        cfg = OmegaConf.load(f"configs/{os.path.basename(spe_folder)}.yaml")
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

    with open(os.path.join("logs", os.path.basename(spe_folder),
            "eval_spe_phase2_res.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
