# Import required packages
import argparse
import os.path as osp

import gym
import mani_skill2.envs
import numpy as np
from mani_skill2.utils.wrappers import RecordEpisode
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder

from utils import ContinuousTaskWrapper, SuccessInfoWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="Config name.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-ids", default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    return args


def single_evaluate(cfg, model_ids, render, model_path):
    if "seed" in cfg.env:
        set_random_seed(cfg.env.seed)

    def make_env(
        env_id: str,
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
    log_dir = osp.join("logs", cfg.trial_name)
    record_dir = osp.join(log_dir, "videos_eval") if render else None
    eval_env = SubprocVecEnv(
        [make_env(cfg.env.name, record_dir=record_dir) for _ in range(cfg.env.n_env_procs)]
    )
    eval_env = VecMonitor(eval_env)  # attach this so SB3 can log reward metrics
    eval_env.seed(cfg.env.seed)
    eval_env.reset()

    model = eval(cfg.model_name)(
        "MlpPolicy",
        eval_env,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        learning_rate=cfg.train.lr,
        tensorboard_log=log_dir,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        **cfg.model_kwargs
    )

    # load model
    if model_path is not None:
        model_path = model_path
        model.set_parameters(model_path)

    # Evaluate the model
    returns, ep_lens = evaluate_policy(
        model,
        eval_env,
        deterministic=True,
        render=render,
        return_episode_rewards=True,
        n_eval_episodes=cfg.eval.n_final_eval_episodes,
    )
    success = np.array(ep_lens) < 200
    success_rate = success.mean()
    print(f"Model {model_ids} success rate: {success_rate}")

    return success_rate


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    single_evaluate(cfg, args.model_ids, args.render, args.model_path)


if __name__ == "__main__":
    main()
