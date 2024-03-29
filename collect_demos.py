import json
import os
from argparse import ArgumentParser
from itertools import chain

import gym
import h5py
import mani_skill2.envs
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3 import PPO, SAC
from tqdm import tqdm

from stable_baselines3.common.utils import set_random_seed


def make_env(env_id, model_id, obs_mode, act_mode):
    env = gym.make(env_id, obs_mode=obs_mode, reward_mode="dense",
        control_mode=act_mode, model_ids=model_id)

    return env


@torch.no_grad()
def run_episode(env, policy):
    obs = env.reset()
    done = False
    observations = [obs]
    actions = []
    rewards = []
    dones = []
    infos = []

    while not done:
        action = policy.predict(obs, deterministic=True)[0]
        actions.append(action)
        obs, reward, done, info = env.step(action)
        dones.append(int(done))
        rewards.append(reward)
        observations.append(obs)
        infos.append(info)

    observations = np.stack(observations)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    dones = np.stack(dones)
    infos = np.array(infos)

    return observations, actions, rewards, dones, infos


def main():
    parser = ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    parser.add_argument("--demos-per-model", default=100, type=int)
    parser.add_argument("--max-trials-per-model", default=500, type=int)

    args = parser.parse_args()
    args.seed = 42 * args.seed
    set_random_seed(args.seed)

    if args.debug:
        args.demos_per_model = 1
        args.max_trials_per_model = 1
    cfg = OmegaConf.load(f"logs/{args.env}/seed={args.seed}/phase1_gen.yaml")
    model_ids = cfg.env.model_ids.split(",")
    if args.by_pcd:
        spe_model_path = f"logs/{args.env}/seed={args.seed}/spe_models_by_pcd.json"
    elif args.by_orc:
        spe_model_path = f"logs/{args.env}/seed={args.seed}/spe_models_by_orc.json"
    else:
        spe_model_path = f"logs/{args.env}/seed={args.seed}/spe_models.json"
        
    with open(spe_model_path, "r") as f:
        spe_model_map = json.load(f)
    spe_model_ids = list(chain(*[v for v in spe_model_map.values()]))
    gen_model_ids = [model_id for model_id in model_ids
        if model_id not in spe_model_ids]
    policy_model_map = {}
    for model_id in gen_model_ids:
        policy_model_map[model_id] = "gen"
    for spe_id, models in spe_model_map.items():
        for model in models:
            policy_model_map[model] = spe_id

    dummy_env = make_env(cfg.env.name, gen_model_ids, cfg.env.obs_mode,
        cfg.env.act_mode)
    policy = eval(cfg.model_name)(
        "MlpPolicy",
        dummy_env,
        batch_size=cfg.train.batch_size,
        gamma=cfg.train.gamma,
        learning_rate=cfg.train.lr,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        **cfg.model_kwargs
    )
    policy.policy.eval()

    if args.by_pcd:
        demo_path = f"logs/{args.env}/seed={args.seed}/demos_by_pcd.h5"
    elif args.by_orc:
        demo_path = f"logs/{args.env}/seed={args.seed}/demos_by_orc.h5"
    else:
        demo_path = f"logs/{args.env}/seed={args.seed}/demos.h5"
    demo_file = h5py.File(demo_path, "w")

    progress = tqdm(total=len(model_ids))
    for model_id in model_ids:
        demo_cnt = 0
        env = make_env(cfg.env.name, model_id, cfg.env.obs_mode,
            cfg.env.act_mode)
        policy_id = policy_model_map[model_id]
        phase = "phase1" if policy_id == "gen" else "phase2"
        if policy_id == "gen" or ((not args.by_pcd) and (not args.by_orc)):
            policy_weight_path = f"logs/{args.env}/seed={args.seed}/{phase}_{policy_id}.zip"
        elif args.by_pcd:
            policy_weight_path = f"logs/{args.env}/seed={args.seed}/{phase}_{policy_id}_by_pcd.zip"
        elif args.by_orc:
            policy_weight_path = f"logs/{args.env}/seed={args.seed}/{phase}_{policy_id}_by_orc.zip"

        policy.set_parameters(policy_weight_path)

        for t in range(args.max_trials_per_model):
            obs, acts, rewards, dones, infos = run_episode(env, policy)

            if infos[-1]["success"]:
                demo_file.create_dataset(f"traj_{model_id}_{demo_cnt}_obs", data=obs)
                demo_file.create_dataset(f"traj_{model_id}_{demo_cnt}_acts", data=acts)
                demo_file.create_dataset(f"traj_{model_id}_{demo_cnt}_rewards", data=rewards)
                demo_file.create_dataset(f"traj_{model_id}_{demo_cnt}_dones", data=dones)
                demo_cnt += 1

            progress.set_description_str(f"{model_id}: {demo_cnt}/{t}/{args.max_trials_per_model}")

            if demo_cnt >= args.demos_per_model:
                break

        progress.update()

    progress.close()

    demo_file.close()


if __name__ == "__main__":
    main()
