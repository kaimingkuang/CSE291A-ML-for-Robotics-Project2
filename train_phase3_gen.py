# Import required packages
import argparse
import os
import os.path as osp
from copy import deepcopy

import gym
import mani_skill2.envs
import numpy as np
import wandb
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

from demo import DemoNpzDataset
from gail import GAILSAC
from utils import ContinuousTaskWrapper, SuccessInfoWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-steps", default=3000000, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.seed = 42 * args.seed
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen/phase1_gen.yaml")
    cfg.model_name = "GAILSAC"
    cfg.train.total_steps = args.n_steps
    if args.by_pcd:
        cfg.trial_name = f"{args.env}_phase3_gen_by_pcd"
    elif args.by_orc:
        cfg.trial_name = f"{args.env}_phase3_gen_by_orc"
    else:
        cfg.trial_name = f"{args.env}_phase3_gen"
    cfg.train.demo_batch_size = cfg.train.batch_size
    OmegaConf.save(config=cfg, f=f"logs/{args.env}/seed={args.seed}/phase3_gen.yaml")

    if not args.debug:
        wandb.login(key="afc534a6cee9821884737295e042db01471fed6a")
        wandb.init(
            entity="kaiming-kuang",
            # set the wandb project where this run will be logged
            project="cse-291a-project2",
            # track hyperparameters and run metadata
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=True
        )
        wandb.run.name = cfg.trial_name
    else:
        cfg.train.total_steps = 10000
        cfg.eval.eval_freq = 1
        cfg.env.n_env_procs = 8
        cfg.eval.n_final_eval_episodes = 10
    
    if args.by_pcd:
        log_dir = f"logs/{args.env}/seed={args.seed}/phase3_gen_by_pcd"
    elif args.by_orc:
        log_dir = f"logs/{args.env}/seed={args.seed}/phase3_gen_by_orc"
    else:
        log_dir = f"logs/{args.env}/seed={args.seed}/phase3_gen"
    os.makedirs(log_dir, exist_ok=True)

    set_random_seed(args.seed)

    def make_env(
        env_id: str,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            env = gym.make(
                env_id,
                obs_mode=cfg.env.obs_mode,
                reward_mode="dense",
                control_mode=cfg.env.act_mode
            )
            # For training, we regard the task as a continuous task with infinite horizon.
            # you can use the ContinuousTaskWrapper here for that
            if max_episode_steps is not None:
                env = ContinuousTaskWrapper(env, max_episode_steps)
            if record_dir is not None:
                env = SuccessInfoWrapper(env)
                env = RecordEpisode(
                    env, record_dir, info_on_video=True, render_mode="cameras"
                )
            return env

        return _init

    # create eval environment
    record_dir = osp.join(log_dir, "videos")
    eval_env = SubprocVecEnv(
        [make_env(cfg.env.name, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(cfg.env.seed)
    eval_env.reset()

    # Create vectorized environments for training
    env = SubprocVecEnv(
        [
            make_env(cfg.env.name, max_episode_steps=cfg.train.max_eps_steps)
            for _ in range(cfg.env.n_env_procs)
        ]
    )
    env = VecMonitor(env)
    env.seed(cfg.env.seed)
    env.reset()

    # demo dataloader
    demo_ds = DemoNpzDataset(f"logs/{args.env}/seed={args.seed}/demos.npz")
    demo_dl = DemoNpzDataset.get_dataloader(demo_ds, cfg.train.demo_batch_size)

    model = eval(cfg.model_name)(
        "MlpPolicy",
        env,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        learning_rate=cfg.train.lr,
        tensorboard_log=log_dir,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        **cfg.model_kwargs
    )

    # load model
    model.set_parameters(f"logs/{args.env}/seed={args.seed}/phase1_gen.zip")

    # define callbacks to periodically save our model and evaluate it to help monitor training
    # the below freq values will save every 10 rollouts
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=cfg.eval.eval_freq * cfg.train.rollout_steps // cfg.env.n_env_procs,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.eval.eval_freq * cfg.train.rollout_steps // cfg.env.n_env_procs,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [checkpoint_callback, eval_callback]
    if not args.debug:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)
    # Train an agent with PPO for args.total_timesteps interactions
    model.learn(
        cfg.train.total_steps,
        callback=callbacks,
        demos=demo_dl
    )
    # Save the final model
    model.save(osp.join(log_dir, "latest_model"))

    # load the best model
    model.set_parameters(osp.join(log_dir, "latest_model.zip"))

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        n_eval_episodes=cfg.eval.n_final_eval_episodes,
    )
    print("Returns", returns)
    print("Episode Lengths", ep_lens)
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print("Success Rate:", success_rate)

    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
