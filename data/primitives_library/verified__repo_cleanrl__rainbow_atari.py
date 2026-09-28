import collections
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 0.0000625
    num_envs: int = 1
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 8000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4
    n_step: int = 3
    prioritized_replay_alpha: float = 0.5
    prioritized_replay_beta: float = 0.4
    prioritized_replay_eps: float = 1e-6
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10


def make_env(env_id, seed, idx, capture_video, run_name):
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
        environment.action_space.seed(seed)
        return environment

    return thunk


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


class NoisyDuelingDistributionalNetwork(nn.Module):
    def __init__(self, env, n_atoms, v_min, v_max):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.n_actions = env.single_action_space.n

        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        features = 3136
        self.value_head = nn.Sequential(
            NoisyLinear(features, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms),
        )
        self.advantage_head = nn.Sequential(
            NoisyLinear(features, 512),
            nn.ReLU(),
            NoisyLinear(512, n_atoms * self.n_actions),
        )

    def forward(self, x):
        features = self.network(x / 255.0)
        value = self.value_head(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_head(features).view(-1, self.n_actions, self.n_atoms)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return F.softmax(logits, dim=2)

    def reset_noise(self):
        for layer in self.value_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_head:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch",
    ["observations", "actions", "rewards", "next_observations", "dones", "indices", "weights"],
)


class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float32)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self):
        return self.tree[0]


class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, alpha, n_step=1, gamma=0.99):
        self.buffer_size = buffer_size
        self.device = device
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma

        self.observations = np.zeros((buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((buffer_size, *observation_space.shape), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, *action_space.shape), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0
        self.sum_tree = SumSegmentTree(buffer_size)
        self.min_tree = MinSegmentTree(buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)

    def _n_step_transition(self):
        reward, next_observation, done = self.n_step_buffer[-1][3:]
        for _, candidate_next_obs, _, candidate_reward, candidate_done in reversed(list(self.n_step_buffer)[:-1]):
            reward = candidate_reward + self.gamma * reward * (1.0 - candidate_done)
            if candidate_done:
                next_observation = candidate_next_obs
                done = candidate_done
        return reward, next_observation, done

    def add(self, obs, next_obs, action, reward, done):
        self.n_step_buffer.append((obs, next_obs, action, reward, done))
        if len(self.n_step_buffer) < self.n_step:
            return

        reward, next_obs, done = self._n_step_transition()
        obs, _, action, _, _ = self.n_step_buffer[0]

        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        priority = self.max_priority**self.alpha
        self.sum_tree.update(self.pos, priority)
        self.min_tree.update(self.pos, priority)

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size, beta):
        total = self.sum_tree.total()
        indices = np.asarray(
            [self.sum_tree.retrieve(random.uniform(0, total)) for _ in range(batch_size)],
            dtype=np.int64,
        )

        probabilities = np.asarray([self.sum_tree.tree[index + self.buffer_size - 1] / total for index in indices])
        min_probability = self.min_tree.min() / total
        max_weight = (self.size * min_probability) ** (-beta)
        weights = (self.size * probabilities) ** (-beta) / max_weight

        return PrioritizedBatch(
            observations=torch.tensor(self.observations[indices], device=self.device),
            actions=torch.tensor(self.actions[indices], device=self.device),
            rewards=torch.tensor(self.rewards[indices], device=self.device),
            next_observations=torch.tensor(self.next_observations[indices], device=self.device),
            dones=torch.tensor(self.dones[indices], device=self.device),
            indices=indices,
            weights=torch.tensor(weights, dtype=torch.float32, device=self.device),
        )

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            priority = float(priority)
            assert priority > 0
            self.max_priority = max(self.max_priority, priority)
            adjusted_priority = priority**self.alpha
            self.sum_tree.update(index, adjusted_priority)
            self.min_tree.update(index, adjusted_priority)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "rainbow_atari.py supports only a single environment"

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
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + index, index, args.capture_video, run_name) for index in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    q_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=1.5e-4)
    target_network = NoisyDuelingDistributionalNetwork(envs, args.n_atoms, args.v_min, args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        args.prioritized_replay_alpha,
        args.n_step,
        args.gamma,
    )

    start_time = time.time()
    observations, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.asarray([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                distributions = q_network(torch.tensor(observations, device=device))
                q_values = (distributions * q_network.support).sum(dim=2)
                actions = q_values.argmax(dim=1).cpu().numpy()

        next_observations, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_observations = next_observations.copy()
        if "final_observation" in infos:
            for index, final_observation in enumerate(infos["final_observation"]):
                if final_observation is not None:
                    real_next_observations[index] = final_observation

        rb.add(
            observations[0],
            real_next_observations[0],
            actions[0],
            rewards[0],
            dones[0],
        )
        observations = next_observations

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            data = rb.sample(args.batch_size, args.prioritized_replay_beta)

            with torch.no_grad():
                target_distribution = target_network(data.next_observations)
                target_q_values = (target_distribution * target_network.support).sum(dim=2)
                next_actions = target_q_values.argmax(dim=1)
                target_distribution = target_distribution[
                    torch.arange(args.batch_size, device=device),
                    next_actions,
                ]

                target_support = data.rewards.unsqueeze(1) + (
                    args.gamma**args.n_step
                ) * (1.0 - data.dones.unsqueeze(1)) * target_network.support.unsqueeze(0)
                target_support = target_support.clamp(args.v_min, args.v_max)

                b = (target_support - args.v_min) / q_network.delta_z
                lower = b.floor().long()
                upper = b.ceil().long()

                projected_distribution = torch.zeros_like(target_distribution)
                offset = (
                    torch.arange(args.batch_size, device=device).unsqueeze(1) * args.n_atoms
                )

                projected_distribution.view(-1).index_add_(
                    0,
                    (lower + offset).view(-1),
                    (target_distribution * (upper.float() - b)).view(-1),
                )
                projected_distribution.view(-1).index_add_(
                    0,
                    (upper + offset).view(-1),
                    (target_distribution * (b - lower.float())).view(-1),
                )

                equal_atoms = lower == upper
                projected_distribution.view(-1).index_add_(
                    0,
                    (lower + offset)[equal_atoms],
                    target_distribution[equal_atoms],
                )

            current_distribution = q_network(data.observations)
            chosen_distribution = current_distribution[
                torch.arange(args.batch_size, device=device),
                data.actions.long().view(-1),
            ].clamp(min=1e-5)

            elementwise_loss = -(projected_distribution * chosen_distribution.log()).sum(dim=1)
            loss = (elementwise_loss * data.weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            priorities = elementwise_loss.detach().cpu().numpy() + args.prioritized_replay_eps
            rb.update_priorities(data.indices, priorities)

            q_network.reset_noise()
            target_network.reset_noise()

            writer.add_scalar("losses/loss", loss.item(), global_step)
            writer.add_scalar("losses/old_val", chosen_distribution.mean().item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            for target_parameter, parameter in zip(target_network.parameters(), q_network.parameters()):
                target_parameter.data.copy_(args.tau * parameter.data + (1.0 - args.tau) * target_parameter.data)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(
                args,
                repo_id,
                "Rainbow Atari",
                model_path,
                args.env_id,
                eval_episodes=10,
                run_name=run_name,
            )

    envs.close()
    writer.close()