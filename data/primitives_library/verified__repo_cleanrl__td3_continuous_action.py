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
    """the name of this experiment"""
    seed: int = 1
    """the seed used for random number generators"""
    torch_deterministic: bool = True
    """whether cuDNN deterministic mode is enabled"""
    cuda: bool = True
    """whether CUDA should be used when available"""
    track: bool = False
    """whether to log the run to Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the Weights and Biases project"""
    wandb_entity: str = None
    """the Weights and Biases entity"""
    capture_video: bool = False
    """whether to record environment videos"""
    save_model: bool = False
    """whether to save learned parameters"""
    upload_model: bool = False
    """whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """the Hugging Face user or organization"""

    env_id: str = "Hopper-v4"
    """the Gymnasium environment identifier"""
    total_timesteps: int = 1000000
    """the total number of environment interactions"""
    learning_rate: float = 3e-4
    """the optimizer learning rate"""
    num_envs: int = 1
    """the number of vectorized environments"""
    buffer_size: int = int(1e6)
    """the replay buffer capacity"""
    gamma: float = 0.99
    """the reward discount"""
    tau: float = 0.005
    """the target-network interpolation factor"""
    batch_size: int = 256
    """the replay minibatch size"""
    policy_noise: float = 0.2
    """the standard deviation used for target policy smoothing"""
    exploration_noise: float = 0.1
    """the action exploration noise multiplier"""
    learning_starts: int = 25e3
    """the interaction step after which learning begins"""
    policy_frequency: int = 2
    """the delayed actor-update interval"""
    noise_clip: float = 0.5
    """the target-policy noise clipping limit"""


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


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(observation_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        joined = torch.cat((x, a), dim=1)
        joined = F.relu(self.fc1(joined))
        joined = F.relu(self.fc2(joined))
        return self.fc3(joined)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(observation_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_size)

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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

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
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
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
                clipped_noise = (
                    torch.randn_like(data.actions, device=device) * args.policy_noise
                ).clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0],
                    envs.single_action_space.high[0],
                )

                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * min_qf_next_target.view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for parameter, target_parameter in zip(actor.parameters(), target_actor.parameters()):
                    target_parameter.data.copy_(args.tau * parameter.data + (1 - args.tau) * target_parameter.data)

                for parameter, target_parameter in zip(qf1.parameters(), qf1_target.parameters()):
                    target_parameter.data.copy_(args.tau * parameter.data + (1 - args.tau) * target_parameter.data)

                for parameter, target_parameter in zip(qf2.parameters(), qf2_target.parameters()):
                    target_parameter.data.copy_(args.tau * parameter.data + (1 - args.tau) * target_parameter.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                print("SPS:", sps)
                writer.add_scalar("charts/SPS", sps, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save((actor.state_dict(), qf1.state_dict(), qf2.state_dict()), model_path)
        print(f"model saved to {model_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(args, repo_name, model_path, f"videos/{run_name}")

    envs.close()
    writer.close()