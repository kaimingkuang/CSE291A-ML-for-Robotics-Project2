import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import wandb
from tqdm import tqdm

rng = np.random.default_rng(0)

env = gym.make("CartPole-v0")
expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
expert.learn(1000)

rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

venv = make_vec_env("CartPole-v0", n_envs=8, rng=rng)
learner = PPO(env=venv, policy=MlpPolicy)
reward_net = BasicRewardNet(
    venv.observation_space,
    venv.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
avg_rew_before = np.mean(rewards)

gail_trainer.train(20000)
rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
avg_rew_after = np.mean(rewards)
print(f"Before={avg_rew_before:.4f}, after={avg_rew_after:.4f}")
