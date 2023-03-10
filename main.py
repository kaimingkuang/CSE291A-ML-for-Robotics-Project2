# Import required packages
import argparse
import os
import os.path as osp

import gym
import mani_skill2.envs
import numpy as np
import wandb
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

from utils import ContinuousTaskWrapper, SuccessInfoWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Config name.")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--name", default="")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = OmegaConf.load(f"configs/{args.cfg}.yaml")
    if args.name != "":
        cfg.trial_name = args.name

    if not args.debug:
        wandb_cfg = OmegaConf.load(f"wandb_cfg.yaml")
        wandb.login(key=wandb_cfg.key)
        wandb.init(
            entity=wandb_cfg.entity,
            # set the wandb project where this run will be logged
            project=wandb_cfg.project,
            # track hyperparameters and run metadata
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=True
        )
        wandb.run.name = cfg.trial_name

    log_dir = f"{cfg.log_dir}/{cfg.trial_name}"
    os.makedirs(log_dir, exist_ok=True)

    if "seed" in cfg.env:
        set_random_seed(cfg.env.seed)

    def make_env(
        env_id: str,
        model_ids=None,
        max_episode_steps: int = None,
        record_dir: str = None,
    ):
        def _init() -> gym.Env:
            # NOTE: Import envs here so that they are registered with gym in subprocesses
            import mani_skill2.envs

            if model_ids is not None:
                env = gym.make(
                    env_id,
                    obs_mode=cfg.env.obs_mode,
                    reward_mode="dense",
                    control_mode=cfg.env.act_mode,
                    model_ids=model_ids.split(",")
                )
            else:
                env = gym.make(
                    env_id,
                    model_idsobs_mode=cfg.env.obs_mode,
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
        [make_env(cfg.env.name, cfg.env.model_ids, record_dir=record_dir) for _ in range(1)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(cfg.env.seed)
    eval_env.reset()

    # Create vectorized environments for training
    env = SubprocVecEnv(
        [
            make_env(cfg.env.name, cfg.env.model_ids, max_episode_steps=cfg.train.max_eps_steps)
            for _ in range(cfg.env.n_env_procs)
        ]
    )
    env = VecMonitor(env)
    env.seed(cfg.env.seed)
    env.reset()

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

    # if args.eval:
    # load model
    if args.model_path is not None:
        model_path = args.model_path
        model.set_parameters(model_path)
    if "init_model_path" in cfg:
        model_path = cfg.init_model_path
        model.set_parameters(model_path)

    # define callbacks to periodically save our model and evaluate it to help monitor training
    # the below freq values will save every 10 rollouts
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10 * cfg.train.rollout_steps // cfg.env.n_env_procs,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10 * cfg.train.rollout_steps // cfg.env.n_env_procs,
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
