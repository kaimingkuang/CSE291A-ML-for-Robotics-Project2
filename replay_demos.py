import argparse
import os

from omegaconf import OmegaConf
from tqdm import tqdm

from demo import DemoDataset, load_h5_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    model_ids = cfg.env.model_ids

    for model_id in tqdm(model_ids):
        os.system(f"python -m mani_skill2.trajectory.replay_trajectory --traj-path \
            demos/rigid_body/TurnFaucet-v0/{model_id}.h5 --save-video --env-id=TurnFaucet-v2 \
            --obs-mode state --target-control-mode pd_ee_delta_pose --num-procs 8")
        for file in os.listdir("demos/rigid_body/TurnFaucet-v0"):
            if file.endswith(".mp4") and "_" not in file:
                os.rename(os.path.join("demos/rigid_body/TurnFaucet-v0", file), os.path.join("demos/rigid_body/TurnFaucet-v0", f"{model_id}_" + file))


if __name__ == "__main__":
    main()
