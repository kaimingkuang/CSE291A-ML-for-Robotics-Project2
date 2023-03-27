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
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gauss-ker-size-pct", default=0.1, type=int,
        help="The size of the Gaussian smoothing kernel.")
    parser.add_argument("--window-size-pct", default=0.1, type=float,
        help="The sliding window size by percentage of training.")
    parser.add_argument("--imp-margin-pct", default=0.05, type=float,
        help="The improvement margin when searching plateaus.")
    parser.add_argument("--plateau-beg-pct", default=0.3, type=float,
        help="The start step to search plateaus by percentage of training.")
    parser.add_argument("--max-steps", default=None, type=int)
    args = parser.parse_args()

    args.seed = 42 * args.seed
    log_dir = f"logs/{args.env}/seed={args.seed}/phase1_gen"
    tb_folders = glob.glob(f"{log_dir}/SAC_*")
    tb_folder = sorted(tb_folders, key=lambda x: int(x.split("_")[-1]))[-1]
    tb_path = glob.glob(f"{tb_folder}/events.*")[0]

    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()
    if args.max_steps is not None:
        train_rewards = np.array([x.value for x
            in ea.Scalars("rollout/ep_rew_mean") if x.step <= args.max_steps])
        train_steps = np.array([x.step for x
            in ea.Scalars("rollout/ep_rew_mean") if x.step <= args.max_steps])
    else:
        train_rewards = np.array([x.value for x
            in ea.Scalars("rollout/ep_rew_mean")])
        train_steps = np.array([x.step for x
            in ea.Scalars("rollout/ep_rew_mean")])

    gauss_ker_size = int(args.gauss_ker_size_pct * train_rewards.shape[0])
    smooth_train_rewards = gaussian_filter1d(train_rewards,
        gauss_ker_size)
    # plt.plot(smooth_train_rewards)
    # plt.savefig("smooth.png")
    window_size = int(smooth_train_rewards.shape[0] * args.window_size_pct)
    start = int(args.plateau_beg_pct * smooth_train_rewards.shape[0])
    plateau_criteria = np.zeros_like(smooth_train_rewards)
    imp_margin = args.imp_margin_pct * (smooth_train_rewards.max() - smooth_train_rewards.min())

    for t in range(start, smooth_train_rewards.shape[0]):
        window_rewards = smooth_train_rewards[t:t + window_size]
        plateau_criteria[t] = (smooth_train_rewards[t] + imp_margin\
            >= window_rewards).mean()

    plateau_step = train_steps[np.min(np.argwhere(plateau_criteria == 1))]
    print(f"The generalist training plateaus at step {plateau_step}/{train_steps.max()}.")
    weight_dir = os.path.dirname(os.path.dirname(tb_path))
    weight_paths = sorted([os.path.join(weight_dir, file)
        for file in os.listdir(weight_dir) if file.startswith("rl_model")],
        key=lambda x: int(x.split("/")[-1].split("_")[2]))
    weight_steps = np.array(sorted([int(file.split("_")[2]) for file
        in os.listdir(weight_dir) if file.startswith("rl_model")]))
    plateau_step = np.argmin(np.abs(weight_steps - plateau_step))
    plateau_weight_path = weight_paths[plateau_step]
    if not args.debug:
        shutil.copyfile(plateau_weight_path, f"logs/{args.env}/seed={args.seed}/phase1_gen.zip")
    print(f"The generalist weights have been saved at logs/{args.env}/seed={args.seed}/phase1_gen.zip")


if __name__ == "__main__":
    main()
