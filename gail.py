import numpy as np
import torch
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac import SAC
from torch.nn import functional as F
from torch.optim import AdamW
import torch.nn as nn


class GAILSAC(SAC):
    """
    SAC-based GAIL.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dis_in_feats = self.policy.actor.latent_pi[0].in_features\
            + self.policy.actor.mu.out_features
        self.discriminator = nn.Sequential(
            nn.Linear(dis_in_feats, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.discriminator.to(self.policy.device)
        self.dis_optimizer = AdamW(self.discriminator.parameters())
        self.dis_loss_fn = nn.BCELoss()
        self.env_reward_proportion = 0.3

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, discriminator_losses = [], [], []

        for gradient_step, demo_batch in zip(range(gradient_steps), self.demos):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # combined the sample data with demos
            demo_obs, demo_acts, demo_rewards, demo_next_obs, demo_dones = demo_batch
            demo_obs = demo_obs.to(replay_data.observations.device)
            demo_acts = demo_acts.to(replay_data.observations.device)
            demo_rewards = demo_rewards.to(replay_data.observations.device)
            demo_next_obs = demo_next_obs.to(replay_data.observations.device)
            demo_dones = demo_dones.to(replay_data.observations.device)
            sampled_batch = {}
            sampled_batch["obs"] = torch.cat([replay_data.observations, demo_obs])
            sampled_batch["actions"] = torch.cat([replay_data.actions, demo_acts])
            sampled_batch["rewards"] = torch.cat([replay_data.rewards, demo_rewards])
            sampled_batch["next_obs"] = torch.cat([replay_data.next_observations, demo_next_obs])
            sampled_batch["dones"] = torch.cat([replay_data.dones, demo_dones])

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(sampled_batch["obs"])
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                # calculate discriminator rewards and combine them with the original rewards
                discriminator_rewards = torch.sigmoid(self.discriminator(torch.cat([sampled_batch["obs"], sampled_batch["actions"]], dim=-1)))
                assert sampled_batch["rewards"].size() == discriminator_rewards.size()
                old_rewards = sampled_batch["rewards"]
                # sampled_batch["rewards"] = self.env_reward_proportion * old_rewards + (1 - self.env_reward_proportion) * discriminator_rewards
                sampled_batch["rewards"] = discriminator_rewards
                clip_max = torch.clamp(old_rewards, min=-0.5)
                sampled_batch["rewards"] = torch.clamp(sampled_batch["rewards"], max=clip_max)

                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(sampled_batch["next_obs"])
                # Compute the next Q values: min over all critics targets
                next_q_values = torch.cat(self.critic_target(sampled_batch["next_obs"], next_actions), dim=1)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = sampled_batch["rewards"] + (1 - sampled_batch["dones"]) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(sampled_batch["obs"], sampled_batch["actions"])

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = torch.cat(self.critic(sampled_batch["obs"], actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # disciminator loss
            self.dis_optimizer.zero_grad()
            replay_bs = replay_data.observations.size(0)
            demo_bs = demo_obs.size(0)
            device = actor_loss.device
            # discriminator labels for demonstrations and the policy
            replay_labels = torch.zeros(replay_bs, 1, device=device, dtype=torch.float)
            demo_labels = torch.ones(demo_bs, 1, device=device, dtype=torch.float)

            # calculate the discriminator loss for demos
            dis_inputs_demo = torch.cat((demo_obs, demo_acts), dim=1).to(device)
            prob_demo = torch.sigmoid(self.discriminator(dis_inputs_demo))
            dis_loss = self.dis_loss_fn(prob_demo, demo_labels)
            # calculate the discriminator loss for the policy
            dis_inputs_policy = torch.cat((replay_data.observations, replay_data.actions), dim=1).to(device)
            prob_policy = torch.sigmoid(self.discriminator(dis_inputs_policy))
            dis_loss += self.dis_loss_fn(prob_policy, replay_labels)

            dis_loss.backward()
            self.dis_optimizer.step()
            discriminator_losses.append(dis_loss.item())

            # Optimize the actor and the discriminator
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/discriminator_loss", np.mean(discriminator_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback,
        log_interval: int = 4,
        tb_log_name: str = "GAIL",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        demos=None
    ):
        assert demos is not None, "Please pass in demos."
        self.demos = demos
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )


if __name__ == "__main__":
    pass
