import gym
import h5py
import numpy as np
from imitation.data.types import TrajectoryWithRew
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EventCallback
from tensorboard.backend.event_processing import event_accumulator


# Defines a continuous, infinite horizon, task where done is always False
# unless a timelimit is reached.
class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps: int) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self._max_episode_steps = max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        return super().reset()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
        else:
            done = False
            info["TimeLimit.truncated"] = False
        return ob, rew, done, info


# A simple wrapper that adds a is_success key which SB3 tracks
class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = info["success"]
        return ob, rew, done, info


def find_best_weight(weight_dir, tb_log):
    succ_rates = np.array([x.value for x
        in tb_log.Scalars("eval/success_rate")])
    max_succ_rate = succ_rates.max()
    max_succ_steps = np.argwhere(succ_rates == max_succ_rate).squeeze(axis=-1)
    mean_ep_lens = np.array([x.value for x
        in tb_log.Scalars("eval/mean_ep_length")])
    best_step = max_succ_steps[np.argmin(mean_ep_lens[max_succ_steps]).squeeze()]
    best_step = [x.step for x
        in tb_log.Scalars("eval/success_rate")][best_step]
    best_weight_path = f"{weight_dir}/rl_model_{best_step}_steps.zip"

    return best_weight_path


def read_tensorboard(tb_path):
    ea = event_accumulator.EventAccumulator(tb_path)
    ea.Reload()

    return ea


def find_best_step(tb_log):
    succ_rates = np.array([x.value for x
        in tb_log.Scalars("eval/success_rate")])
    max_succ_rate = succ_rates.max()
    max_succ_steps = np.argwhere(succ_rates == max_succ_rate).squeeze(axis=-1)
    mean_ep_lens = np.array([x.value for x
        in tb_log.Scalars("eval/mean_ep_length")])
    best_step = max_succ_steps[np.argmin(mean_ep_lens[max_succ_steps]).squeeze()]
    best_step = [x.step for x
        in tb_log.Scalars("eval/success_rate")][best_step]

    return best_step
