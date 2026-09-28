import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


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

    env_id: str = "Hopper-v4"
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    learning_starts: int = 5e3
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            environment = gym.make(env_id, render_mode="rgb_array")
            environment = gym.wrappers.RecordVideo(environment, f"videos/{run_name}")
        else:
            environment = gym.make(env_id)
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment.action_space.seed(seed)
        return environment

    return thunk


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(observation_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        features = torch.cat((x, a), dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc3(features)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)

        self.fc1 = nn.Linear(observation_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_size)
        self.fc_logstd = nn.Linear(256, action_size)

        low = env.single_action_space.low
        high = env.single_action_space.high
        self.register_buffer(
            "action_scale",
            torch.tensor((high - low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((high + low) / 2.0, dtype=torch.float32),
        )

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(hidden))
        mean = self.fc_mean(hidden)
        raw_log_std = self.fc_logstd(hidden)
        bounded_log_std = torch.tanh(raw_log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (bounded_log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent_action = distribution.rsample()
        squashed_action = torch.tanh(latent_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = distribution.log_prob(latent_action)
        log_prob = log_prob - torch.log(self.action_scale * (1 - squashed_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, deterministic_action


if __name__ == "__main__":
    args = tyro.cli(Args)
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
        [
            make_env(args.env_id, args.seed + index, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, device=device, requires_grad=True)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episodic_return = info["episode"]["r"]
                    episodic_length = info["episode"]["l"]
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)
                    break

        real_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[index] = infos["final_observation"][index]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions, next_log_pi, _ = actor.get_action(data.next_observations)
                target_q1 = qf1_target(data.next_observations, next_actions)
                target_q2 = qf2_target(data.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2) - alpha * next_log_pi
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * target_q.view(-1)

            qf1_values = qf1(data.observations, data.actions).view(-1)
            qf2_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    policy_actions, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, policy_actions)
                    qf2_pi = qf2(data.observations, policy_actions)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy).detach()).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(qf1_target.parameters(), qf1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for target_param, param in zip(qf2_target.parameters(), qf2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()