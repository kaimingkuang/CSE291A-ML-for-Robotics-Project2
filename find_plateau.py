import glob
import os
import shutil
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing import event_accumulator


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--gauss-ker-size", default=400, type=int,
        help="The size of the Gaussian smoothing kernel.")
    parser.add_argument("--window-size-pct", default=0.1, type=float,
        help="The sliding window size by percentage of training.")
    parser.add_argument("--imp-margin", default=50, type=float,
        help="The improvement margin when searching plateaus.")
    parser.add_argument("--plateau-beg-pct", default=0.15, type=float,
        help="The start step to search plateaus by percentage of training.")
    args = parser.parse_args()

    log_dir = f"logs/{args.env}/phase1_gen"
    tb_folders = glob.glob(f"{log_dir}/SAC_*")
    tb_folder = sorted(tb_folders, key=lambda x: int(x.split("_")[-1]))[-1]
    tb_path = glob.glob(f"{tb_folder}/events.*")[0]

    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    train_rewards = np.array([x.value for x
        in ea.Scalars("rollout/ep_rew_mean")])
    smooth_train_rewards = gaussian_filter1d(train_rewards,
        args.gauss_ker_size)
    window_size = int(smooth_train_rewards.shape[0] * args.window_size_pct)
    start = int(args.plateau_beg_pct * smooth_train_rewards.shape[0])
    plateau_criteria = np.zeros_like(smooth_train_rewards)

    for t in range(start, smooth_train_rewards.shape[0]):
        window_rewards = smooth_train_rewards[t:t + window_size]
        plateau_criteria[t] = (smooth_train_rewards[t] + args.imp_margin\
            >= window_rewards).mean()

    plateau_step = np.min(np.argwhere(plateau_criteria == 1))
    weight_dir = os.path.dirname(os.path.dirname(tb_path))
    weight_paths = [os.path.join(weight_dir, file)
        for file in os.listdir(weight_dir) if file.startswith("rl_model")]
    plateau_step = int(plateau_step / plateau_criteria.shape[0]\
        * len(weight_paths))
    plateau_weight_path = weight_paths[plateau_step]
    print(f"The generalist training plateaus at step {plateau_step}/{len(weight_paths)}.")
    shutil.copyfile(plateau_weight_path, f"logs/{args.env}/phase1_gen.zip")
    print(f"The generalist weights have been saved at logs/{args.env}/phase1_gen.zip")


if __name__ == "__main__":
    main()
