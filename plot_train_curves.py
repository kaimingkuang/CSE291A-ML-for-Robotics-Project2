import glob
import os
import shutil
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils import find_best_step, read_tensorboard


def get_latest_tb(folder, tag):
    tb_folders = glob.glob(f"{folder}/{tag}_*")
    tb_folder = sorted(tb_folders, key=lambda x: int(x.split("_")[-1]))[-1]
    tb_path = glob.glob(f"{tb_folder}/events.*")[0]

    return tb_path


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--no-phase3", action="store_true")
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20000000)
    parser.add_argument("--gauss-ker-size-pct", default=0.1, type=int,
        help="The size of the Gaussian smoothing kernel.")
    parser.add_argument("--window-size-pct", default=0.1, type=float,
        help="The sliding window size by percentage of training.")
    parser.add_argument("--imp-margin-pct", default=0.05, type=float,
        help="The improvement margin when searching plateaus.")
    parser.add_argument("--plateau-beg-pct", default=0.3, type=float,
        help="The start step to search plateaus by percentage of training.")
    args = parser.parse_args()
    args.seed = 42 * args.seed
    log_dir = f"logs/{args.env}/seed={args.seed}"

    sns.set_theme()

    phase1_tb_paths = [get_latest_tb(f"logs/{args.env}/seed={seed}/phase1_gen", "SAC") for seed in [0, 42, 84]]
    phase1_tbs = [read_tensorboard(path)
        for path in phase1_tb_paths]
    phase1_tb = phase1_tbs[1]
    phase1_data = [x.Scalars("rollout/ep_rew_mean") for x in phase1_tbs]
    phase1_data = [pd.DataFrame({"step": [x.step for x in data], "reward": [x.value for x in data]}) for data in phase1_data]
    phase1_data = [x.loc[x.step <= args.max_steps] for x in phase1_data]
    total = deepcopy(phase1_data[0])
    total.rename({"reward": "reward_0"}, axis=1, inplace=True)
    for i in range(1, len(phase1_data)):
        phase1_data[i].rename({"reward": f"reward_{i}"}, axis=1, inplace=True)
        total = total.merge(phase1_data[i], on="step", how="outer")
    total.fillna(method="backfill", inplace=True)
    total.drop_duplicates(["step"], inplace=True, ignore_index=True)
    total["reward"] = total[[f"reward_{i}" for i in range(len(phase1_data))]].values.mean(axis=1)
    phase1_data = total

    if args.by_pcd:
        phase2_tb_paths = glob.glob(f"{log_dir}/phase2_spe*_by_pcd/")
    elif args.by_orc:
        phase2_tb_paths = glob.glob(f"{log_dir}/phase2_spe*_by_orc/")
    else:
        phase2_tb_paths = glob.glob(f"{log_dir}/phase2_spe*/")
        phase2_tb_paths = [(x) for x in phase2_tb_paths if "by" not in x]
    phase2_tb_paths = [get_latest_tb(path, "SAC") for path in phase2_tb_paths]
    phase2_tbs = [read_tensorboard(path) for path in phase2_tb_paths]

    if not args.no_phase3:
        if args.by_pcd:
            phase3_tb_path = get_latest_tb(f"{log_dir}/phase3_gen_by_pcd", "GAIL")
        elif args.by_orc:
            phase3_tb_path = get_latest_tb(f"{log_dir}/phase3_gen_by_orc", "GAIL")
        else:
            phase3_tb_path = get_latest_tb(f"{log_dir}/phase3_gen", "GAIL")
        phase3_tb = read_tensorboard(phase3_tb_path)

    # calculate where is the plateau
    if args.max_steps is not None:
        train_rewards = np.array([x.value for x
            in phase1_tb.Scalars("rollout/ep_rew_mean") if x.step <= args.max_steps])
        train_steps = np.array([x.step for x
            in phase1_tb.Scalars("rollout/ep_rew_mean") if x.step <= args.max_steps])
    else:
        train_rewards = np.array([x.value for x
            in phase1_tb.Scalars("rollout/ep_rew_mean")])
        train_steps = np.array([x.step for x
            in phase1_tb.Scalars("rollout/ep_rew_mean")])

    gauss_ker_size = int(args.gauss_ker_size_pct * train_rewards.shape[0])
    smooth_train_rewards = gaussian_filter1d(train_rewards,
        gauss_ker_size)
    window_size = int(smooth_train_rewards.shape[0] * args.window_size_pct)
    start = int(args.plateau_beg_pct * smooth_train_rewards.shape[0])
    plateau_criteria = np.zeros_like(smooth_train_rewards)
    imp_margin = args.imp_margin_pct * (smooth_train_rewards.max() - smooth_train_rewards.min())

    for t in range(start, smooth_train_rewards.shape[0]):
        window_rewards = smooth_train_rewards[t:t + window_size]
        plateau_criteria[t] = (smooth_train_rewards[t] + imp_margin\
            >= window_rewards).mean()

    plateau_step = train_steps[np.min(np.argwhere(plateau_criteria == 1))]

    # phase 2
    phase2_data = {f"reward_{i}": [x.value for x in phase2_tbs[i].Scalars("rollout/ep_rew_mean")]
        for i in range(len(phase2_tbs))}
    phase2_data["step"] = [x.step + plateau_step for x in phase2_tbs[0].Scalars("rollout/ep_rew_mean")]
    phase2_data = pd.DataFrame(phase2_data)
    phase2_data["reward"] = phase2_data[[f"reward_{i}" for i in range(len(phase2_tbs))]].values.mean(axis=1)

    if not args.no_phase3:
        phase3_best_step = find_best_step(phase3_tb)
        phase3_steps = np.array([x.step for x in phase3_tb.Scalars("rollout/ep_rew_mean")])
        phase3_rewards = np.array([x.value for x in phase3_tb.Scalars("rollout/ep_rew_mean")])
        phase3_best_step_idx = np.argmin([np.abs(step - phase3_best_step) for step in phase3_steps])
        phase3_best_step = phase3_steps[phase3_best_step_idx]
        phase3_best_reward = phase3_rewards[phase3_best_step_idx]
        phase3_data = pd.DataFrame({
            "step": [plateau_step, phase3_best_step + plateau_step],
            "reward": [phase1_data.loc[phase1_data.step <= plateau_step, "reward"].values[-1], phase3_best_reward]
        })

    # phase 1 plot
    sns.lineplot(phase1_data, x="step", y="reward", color="silver")

    # phase 2 plot
    sns.lineplot(phase2_data, x="step", y="reward", color="darkorange")

    # phase 3 plot
    if not args.no_phase3:
        sns.lineplot(phase3_data, x="step", y="reward", color="#c565c7")

    if args.by_pcd:
        plot_path = f"{args.env}_seed={args.seed}_by_pcd.png"
    elif args.by_orc:
        plot_path = f"{args.env}_seed={args.seed}_by_orc.png"
    else:
        plot_path = f"{args.env}_seed={args.seed}.png"
    plt.savefig(plot_path, dpi=600)


if __name__ == "__main__":
    main()
