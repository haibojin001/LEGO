import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.splitext(os.path.basename(__file__))[0]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "Breakout-v5"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    num_minibatches: int = 4
    update_epochs: int = 4
    max_grad_norm: float = 10.0
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    q_lambda: float = 0.65

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return result

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        n_actions = env.single_action_space.n
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
            layer_init(nn.Linear(512, n_actions)),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(start_e + slope * t, end_e)


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
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    avg_returns = deque(maxlen=20)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )

            random_actions = torch.randint(
                envs.single_action_space.n,
                (args.num_envs,),
                device=device,
            )

            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = q_values.argmax(dim=1)
                values[step] = q_values[torch.arange(args.num_envs, device=device), max_actions]

            exploration_mask = torch.rand(args.num_envs, device=device) < epsilon
            action = torch.where(exploration_mask, random_actions, max_actions)
            actions[step] = action

            next_obs_np, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            for idx, terminated in enumerate(next_done):
                if terminated and info["lives"][idx] == 0:
                    episodic_return = info["r"][idx]
                    episodic_length = info["l"][idx]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    avg_returns.append(episodic_return)
                    writer.add_scalar("charts/avg_episodic_return", np.mean(avg_returns), global_step)

        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            next_q_values = q_network(next_obs).max(dim=1).values

            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    continuation = 1.0 - next_done
                    bootstrap_value = next_q_values
                    returns[step] = rewards[step] + args.gamma * continuation * bootstrap_value
                else:
                    continuation = 1.0 - dones[step + 1]
                    bootstrap_value = (1.0 - args.q_lambda) * values[step + 1] + args.q_lambda * returns[step + 1]
                    returns[step] = rewards[step] + args.gamma * continuation * bootstrap_value

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1).long()
        b_returns = returns.reshape(-1)

        indices = np.arange(args.batch_size)
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = indices[start:end]

                predicted_q = q_network(b_obs[minibatch_indices])
                predicted_q = predicted_q.gather(1, b_actions[minibatch_indices].unsqueeze(1)).squeeze(1)
                loss = F.mse_loss(predicted_q, b_returns[minibatch_indices])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred = values.detach().cpu().numpy().flatten()
        y_true = returns.detach().cpu().numpy().flatten()
        variance_y = np.var(y_true)
        explained_variance = np.nan if variance_y == 0 else 1 - np.var(y_true - y_pred) / variance_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()