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
from huggingface_hub import hf_hub_download
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from cleanrl.dqn_atari import QNetwork as TeacherModel
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.evals.dqn_eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
    torch_deterministic: bool = True
    """whether cudnn should use deterministic algorithms"""
    cuda: bool = True
    """whether CUDA should be enabled when available"""
    track: bool = False
    """whether to track the experiment with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the Weights and Biases project name"""
    wandb_entity: str = None
    """the Weights and Biases entity"""
    capture_video: bool = False
    """whether videos should be recorded"""
    save_model: bool = False
    """whether to save the resulting model"""
    upload_model: bool = False
    """whether to upload the resulting model to Hugging Face"""
    hf_entity: str = ""
    """the Hugging Face account or organization name"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the environment identifier"""
    total_timesteps: int = 10000000
    """number of online environment steps"""
    learning_rate: float = 1e-4
    """optimizer learning rate"""
    num_envs: int = 1
    """number of parallel environments"""
    buffer_size: int = 1000000
    """replay-buffer capacity"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 1.0
    """target-network interpolation coefficient"""
    target_network_frequency: int = 1000
    """target-network update period"""
    batch_size: int = 32
    """minibatch size"""
    start_e: float = 1.0
    """initial exploration probability"""
    end_e: float = 0.01
    """final exploration probability"""
    exploration_fraction: float = 0.10
    """portion of training used for epsilon annealing"""
    learning_starts: int = 80000
    """online step at which optimization begins"""
    train_frequency: int = 4
    """optimization period"""

    teacher_policy_hf_repo: str = None
    """repository containing the teacher checkpoint"""
    teacher_model_exp_name: str = "dqn_atari"
    """experiment name used by the teacher checkpoint"""
    teacher_eval_episodes: int = 10
    """episodes used to evaluate the teacher"""
    teacher_steps: int = 500000
    """teacher interactions used to fill replay memory"""
    offline_steps: int = 500000
    """student updates performed before online interaction"""
    temperature: float = 1.0
    """distillation softmax temperature"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            result = gym.make(env_id, render_mode="rgb_array")
            result = gym.wrappers.RecordVideo(result, f"videos/{run_name}")
        else:
            result = gym.make(env_id)

        result = gym.wrappers.RecordEpisodeStatistics(result)
        result = NoopResetEnv(result, noop_max=30)
        result = MaxAndSkipEnv(result, skip=4)
        result = EpisodicLifeEnv(result)
        if "FIRE" in result.unwrapped.get_action_meanings():
            result = FireResetEnv(result)
        result = ClipRewardEnv(result)
        result = gym.wrappers.ResizeObservation(result, (84, 84))
        result = gym.wrappers.GrayScaleObservation(result)
        result = gym.wrappers.FrameStack(result, 4)
        result.action_space.seed(seed)
        return result

    return thunk


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + residual


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(input_shape[0], out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _, height, width = self._input_shape
        return self._out_channels, (height + 1) // 2, (width + 1) // 2


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        channels, height, width = envs.single_observation_space.shape
        current_shape = (channels, height, width)
        layers = []

        for channels in (16, 32, 32):
            sequence = ConvSequence(current_shape, channels)
            current_shape = sequence.get_output_shape()
            layers.append(sequence)

        layers.extend(
            [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(current_shape[0] * current_shape[1] * current_shape[2], 256),
                nn.ReLU(),
                nn.Linear(256, env.single_action_space.n),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    increment = (end_e - start_e) / duration
    return max(start_e + increment * t, end_e)


def kl_divergence_with_logits(target_logits, prediction_logits):
    """Implementation of on-policy distillation loss."""
    target_distribution = F.softmax(target_logits, dim=-1)
    differences = F.log_softmax(prediction_logits, dim=-1) - F.log_softmax(target_logits, dim=-1)
    return torch.sum(-target_distribution * differences)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"

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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    teacher_model_path = hf_hub_download(
        repo_id=args.teacher_policy_hf_repo,
        filename=f"{args.teacher_model_exp_name}.cleanrl_model",
    )
    teacher_model = TeacherModel(envs).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()

    teacher_returns = evaluate(
        teacher_model,
        make_env,
        args.env_id,
        args.teacher_eval_episodes,
        run_name,
        device,
        capture_video=args.capture_video,
        epsilon=0.05,
    )
    for index, episodic_return in enumerate(teacher_returns):
        print(f"teacher_eval_episode={index}, episodic_return={episodic_return}")
        writer.add_scalar("charts/teacher_eval_return", episodic_return, index)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)

    for _ in track(range(args.teacher_steps), description="Generating teacher data"):
        with torch.no_grad():
            actions = teacher_model(torch.Tensor(obs).to(device)).argmax(dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(f"teacher_step={_}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/teacher_episodic_return", info["episode"]["r"], _)
                    writer.add_scalar("charts/teacher_episodic_length", info["episode"]["l"], _)

        stored_next_obs = next_obs.copy()
        for env_index, was_truncated in enumerate(truncations):
            if was_truncated:
                stored_next_obs[env_index] = infos["final_observation"][env_index]

        rb.add(obs, stored_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

    for offline_step in track(range(args.offline_steps), description="Offline student training"):
        data = rb.sample(args.batch_size)

        with torch.no_grad():
            next_values = target_network(data.next_observations).max(dim=1)[0]
            target = data.rewards.flatten() + args.gamma * next_values * (1 - data.dones.flatten())

        values = q_network(data.observations).gather(1, data.actions).squeeze()
        td_loss = F.mse_loss(values, target)

        with torch.no_grad():
            teacher_logits = teacher_model(data.observations)
        student_logits = q_network(data.observations)
        kl_loss = kl_divergence_with_logits(
            teacher_logits / args.temperature,
            student_logits / args.temperature,
        )
        loss = td_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("losses/offline_td_loss", td_loss.item(), offline_step)
        writer.add_scalar("losses/offline_kl_loss", kl_loss.item(), offline_step)
        writer.add_scalar("losses/offline_loss", loss.item(), offline_step)

        if offline_step % args.target_network_frequency == 0:
            for target_parameter, parameter in zip(target_network.parameters(), q_network.parameters()):
                target_parameter.data.copy_(
                    args.tau * parameter.data + (1.0 - args.tau) * target_parameter.data
                )

    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()
    episode_returns = deque(maxlen=100)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = q_network(torch.Tensor(obs).to(device)).argmax(dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    episodic_return = info["episode"]["r"]
                    episodic_length = info["episode"]["l"]
                    episode_returns.append(episodic_return)
                    print(f"global_step={global_step}, episodic_return={episodic_return}")
                    writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                    writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        stored_next_obs = next_obs.copy()
        for env_index, was_truncated in enumerate(truncations):
            if was_truncated:
                stored_next_obs[env_index] = infos["final_observation"][env_index]

        rb.add(obs, stored_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_values = target_network(data.next_observations).max(dim=1)[0]
                    target = data.rewards.flatten() + args.gamma * next_values * (1 - data.dones.flatten())

                values = q_network(data.observations).gather(1, data.actions).squeeze()
                td_loss = F.mse_loss(values, target)

                with torch.no_grad():
                    teacher_logits = teacher_model(data.observations)
                student_logits = q_network(data.observations)
                kl_loss = kl_divergence_with_logits(
                    teacher_logits / args.temperature,
                    student_logits / args.temperature,
                )
                loss = td_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("losses/td_loss", td_loss.item(), global_step)
                writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
                writer.add_scalar("losses/loss", loss.item(), global_step)
                writer.add_scalar("losses/q_values", values.mean().item(), global_step)

            if global_step % args.target_network_frequency == 0:
                for target_parameter, parameter in zip(target_network.parameters(), q_network.parameters()):
                    target_parameter.data.copy_(
                        args.tau * parameter.data + (1.0 - args.tau) * target_parameter.data
                    )

        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(args, repo_id, model_path, args.teacher_eval_episodes, run_name)

    envs.close()
    writer.close()