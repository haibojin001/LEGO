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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observation

    def step(self, action):
        observation, reward, done, info = super().step(action)
        self.episode_returns += info["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - info["terminated"]
        self.episode_lengths *= 1 - info["terminated"]
        info["r"] = self.returned_episode_returns
        info["l"] = self.returned_episode_lengths
        return observation, reward, done, info


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.LayerNorm([32, 20, 20]),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.LayerNorm([64, 9, 9]),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.LayerNorm([64, 7, 7]),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.LayerNorm(512),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 128)
        for parameter_name, parameter in self.lstm.named_parameters():
            if "bias" in parameter_name:
                nn.init.constant_(parameter, 0)
            elif "weight" in parameter_name:
                nn.init.orthogonal_(parameter, 1.0)
        self.q_func = layer_init(nn.Linear(128, env.single_action_space.n))

    def get_states(self, x, lstm_state, done):
        features = self.network(x / 255.0)
        batch_size = lstm_state[0].shape[1]
        features = features.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        outputs = []

        for feature, reset_mask in zip(features, done):
            output, lstm_state = self.lstm(
                feature.unsqueeze(0),
                (
                    (1.0 - reset_mask).view(1, -1, 1) * lstm_state[0],
                    (1.0 - reset_mask).view(1, -1, 1) * lstm_state[1],
                ),
            )
            outputs.append(output)

        return torch.flatten(torch.cat(outputs), 0, 1), lstm_state

    def forward(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        return self.q_func(hidden), lstm_state


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
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
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
        stack_num=1,
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

    next_lstm_state = (
        torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size, device=device),
        torch.zeros(q_network.lstm.num_layers, args.num_envs, q_network.lstm.hidden_size, device=device),
    )

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                q_values, next_lstm_state = q_network(next_obs, next_lstm_state, next_done)
                values[step] = q_values.max(dim=1).values
                epsilon = linear_schedule(
                    args.start_e,
                    args.end_e,
                    args.exploration_fraction * args.total_timesteps,
                    global_step,
                )
                random_actions = torch.randint(
                    envs.single_action_space.n,
                    q_values.shape,
                    device=device,
                )
                greedy_actions = q_values.argmax(dim=1)
                actions[step] = torch.where(torch.rand(q_values.shape[0], device=device) < epsilon, random_actions, greedy_actions)

            next_obs_np, reward, done, infos = envs.step(actions[step].cpu().numpy())
            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            for env_index, terminated in enumerate(infos["terminated"]):
                if terminated:
                    episodic_return = infos["r"][env_index]
                    episodic_length = infos["l"][env_index]
                    avg_returns.append(episodic_return)
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        with torch.no_grad():
            next_q_values, _ = q_network(next_obs, next_lstm_state, next_done)
            next_values = next_q_values.max(dim=1).values
            returns = torch.zeros_like(rewards)

            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    returns[step] = rewards[step] + args.gamma * next_values * (1.0 - next_done)
                else:
                    returns[step] = rewards[step] + args.gamma * (
                        (1.0 - args.q_lambda) * values[step + 1] + args.q_lambda * returns[step + 1]
                    ) * (1.0 - dones[step + 1])

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_dones = dones.reshape(-1)
        b_returns = returns.reshape(-1)
        batch_indices = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices.T)

            for start in range(0, args.num_envs, args.num_envs // args.num_minibatches):
                minibatch_indices = batch_indices[:, start : start + args.num_envs // args.num_minibatches]
                initial_indices = minibatch_indices[0] % args.num_envs
                minibatch_lstm_state = (
                    initial_lstm_state[0][:, initial_indices],
                    initial_lstm_state[1][:, initial_indices],
                )

                predicted_q_values, _ = q_network(
                    b_obs[minibatch_indices],
                    minibatch_lstm_state,
                    b_dones[minibatch_indices],
                )
                predicted_q_values = predicted_q_values.gather(
                    1,
                    b_actions[minibatch_indices].long().reshape(-1, 1),
                ).squeeze(1)

                loss = F.mse_loss(predicted_q_values, b_returns[minibatch_indices].reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss.item(), global_step)
        writer.add_scalar("losses/q_values", predicted_q_values.mean().item(), global_step)
        if len(avg_returns) > 0:
            writer.add_scalar("charts/avg_episodic_return", np.mean(avg_returns), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()