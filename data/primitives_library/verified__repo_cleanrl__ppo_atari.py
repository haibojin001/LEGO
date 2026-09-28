import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    """the name of the experiment"""
    seed: int = 1
    """random seed"""
    torch_deterministic: bool = True
    """whether to configure cuDNN deterministically"""
    cuda: bool = True
    """whether CUDA should be used when available"""
    track: bool = False
    """whether to log the run to Weights & Biases"""
    wandb_project_name: str = "cleanRL"
    """Weights & Biases project name"""
    wandb_entity: str = None
    """Weights & Biases entity"""
    capture_video: bool = False
    """whether to record videos"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """environment identifier"""
    total_timesteps: int = 10000000
    """total environment interactions"""
    learning_rate: float = 2.5e-4
    """optimizer learning rate"""
    num_envs: int = 8
    """number of concurrent environments"""
    num_steps: int = 128
    """rollout length per environment"""
    anneal_lr: bool = True
    """whether to linearly anneal the learning rate"""
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    num_minibatches: int = 4
    """number of minibatches per PPO epoch"""
    update_epochs: int = 4
    """number of PPO optimization epochs"""
    norm_adv: bool = True
    """whether to standardize advantages"""
    clip_coef: float = 0.1
    """PPO clipping threshold"""
    clip_vloss: bool = True
    """whether to use clipped value loss"""
    ent_coef: float = 0.01
    """entropy regularization coefficient"""
    vf_coef: float = 0.5
    """value loss coefficient"""
    max_grad_norm: float = 0.5
    """gradient norm limit"""
    target_kl: float = None
    """optional approximate KL stopping limit"""

    batch_size: int = 0
    """computed rollout batch size"""
    minibatch_size: int = 0
    """computed minibatch size"""
    num_iterations: int = 0
    """computed number of updates"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            environment = gym.make(env_id, render_mode="rgb_array")
            environment = gym.wrappers.RecordVideo(environment, f"videos/{run_name}")
        else:
            environment = gym.make(env_id)

        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment = NoopResetEnv(environment, noop_max=30)
        environment = MaxAndSkipEnv(environment, skip=4)
        environment = EpisodicLifeEnv(environment)

        if "FIRE" in environment.unwrapped.get_action_meanings():
            environment = FireResetEnv(environment)

        environment = ClipRewardEnv(environment)
        environment = gym.wrappers.ResizeObservation(environment, (84, 84))
        environment = gym.wrappers.GrayScaleObservation(environment)
        environment = gym.wrappers.FrameStack(environment, 4)
        return environment

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        features = self.network(x / 255.0)
        distribution = Categorical(logits=self.actor(features))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(features)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]

                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                )

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        batch_indices = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[minibatch_indices],
                    b_actions.long()[minibatch_indices],
                )
                logratio = newlogprob - b_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                minibatch_advantages = b_advantages[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                pg_loss1 = -minibatch_advantages * ratio
                pg_loss2 = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    value_loss_unclipped = (newvalue - b_returns[minibatch_indices]) ** 2
                    value_clipped = b_values[minibatch_indices] + torch.clamp(
                        newvalue - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_loss_clipped = (value_clipped - b_returns[minibatch_indices]) ** 2
                    v_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance_y = np.var(y_true)
        explained_var = np.nan if variance_y == 0 else 1 - np.var(y_true - y_pred) / variance_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()