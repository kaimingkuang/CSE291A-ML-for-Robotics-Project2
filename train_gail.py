import argparse
import json
import pickle
from itertools import chain
from time import perf_counter

import gym
import mani_skill2.envs
import numpy as np
import wandb
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from omegaconf import OmegaConf
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
from wandb.integration.sb3 import WandbCallback

from demo import load_demos


def make_env(env_id, model_id, obs_mode, act_mode):
    env = gym.make(env_id, obs_mode=obs_mode, reward_mode="dense",
        control_mode=act_mode, model_ids=model_id)

    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n-steps", default=3000000, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--by-pcd", action="store_true")
    parser.add_argument("--by-orc", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--demos-per-model", default=20, type=int)
    parser.add_argument("--max-trials-per-model", default=500, type=int)
    parser.add_argument("--fixed-horizon", default=None, type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    args.seed = 42 * args.seed
    rng = np.random.default_rng(0)

    # collect demos
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

    progress = tqdm(total=len(model_ids))
    rollouts = []
    for model_id in model_ids:
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

        policy = eval(cfg.model_name)(
            "MlpPolicy",
            env,
            batch_size=cfg.train.batch_size,
            gamma=cfg.train.gamma,
            learning_rate=cfg.train.lr,
            policy_kwargs={"net_arch": list(cfg.net_arch)},
            **cfg.model_kwargs
        )
        policy.policy.eval()
        policy.set_parameters(policy_weight_path)

        rollouts += rollout.rollout(
            policy,
            make_vec_env(
                args.env,
                n_envs=cfg.env.n_env_procs,
                post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                rng=rng,
                env_make_kwargs={"obs_mode": "state",
                "control_mode": "pd_ee_delta_pose", "reward_mode": "dense"}
            ),
            rollout.make_sample_until(min_timesteps=None, min_episodes=args.demos_per_model),
            rng=rng,
        )

        progress.update()

        if args.debug:
            break

    with open(f"logs/{args.env}/seed={args.seed}/demos.pickle", "wb") as f:
        pickle.dump(rollouts, f)
    progress.close()

    if not args.debug:
        wandb.login(key="afc534a6cee9821884737295e042db01471fed6a")
        wandb.init(
            entity="kaiming-kuang",
            # set the wandb project where this run will be logged
            project="cse-291a-project2",
            # track hyperparameters and run metadata
            sync_tensorboard=True,
            monitor_gym=True
        )
        if args.pretrained:
            wandb.run.name = f"{args.env}_{args.seed}_gail_pretrained"
        else:
            wandb.run.name = f"{args.env}_{args.seed}_gail"

    n_envs = 32 if not args.debug else 8
    venv = make_vec_env(args.env, n_envs=n_envs, rng=rng,
        env_make_kwargs={"obs_mode": "state",
        "control_mode": "pd_ee_delta_pose", "reward_mode": "dense"})
    eval_env = make_vec_env(args.env, n_envs=n_envs, rng=rng,
        env_make_kwargs={"obs_mode": "state",
        "control_mode": "pd_ee_delta_pose", "reward_mode": "dense"})
    if args.pretrained:
        log_dir = f"logs/{args.env}/seed={args.seed}/gail_pretrained"
    else:
        log_dir = f"logs/{args.env}/seed={args.seed}/gail"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=10,
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = [checkpoint_callback, eval_callback]
    if not args.debug:
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)
    
    learner = SAC("MlpPolicy", venv, tensorboard_log=log_dir)

    if args.pretrained:
        learner.set_parameters(f"logs/{args.env}/seed={args.seed}/phase1_gen.zip")

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
        allow_variable_horizon=args.fixed_horizon is None,
    )
    gail_trainer.gen_callback = callbacks

    total_steps = 20000
    update_freq = 100
    n_eval_eps = 20
    n_epochs = total_steps // update_freq

    progress = tqdm(total=n_epochs)
    for i in range(n_epochs):
        gail_trainer.train(update_freq)
        rewards, ep_lens = evaluate_policy(learner, venv, n_eval_eps,
            return_episode_rewards=True)
        avg_rew = np.mean(rewards)
        avg_succ = np.mean([x < 200 for x in ep_lens])
        progress.set_description_str(f"Epoch {i}: rew={avg_rew:.4f}, "\
            f"succ={avg_succ:.4f}")
        progress.update()

        if not args.debug:
            wandb.log({
                "epoch": i,
                "global_steps": i * update_freq,
                "avg_reward": avg_rew,
                "avg_success_rate": avg_succ
            })

        print(f"Epoch {i}: rew={avg_rew:.4f}, succ={avg_succ:.4f}")

    progress.close()
    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
