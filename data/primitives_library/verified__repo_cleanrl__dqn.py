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
    exp_name: str = os.path.basename(__file__)[:-3]
    """Experiment label."""
    seed: int = 1
    """Random seed."""
    torch_deterministic: bool = True
    """Whether cuDNN should operate deterministically."""
    cuda: bool = True
    """Whether CUDA may be selected when it is available."""
    track: bool = False
    """Whether to log the run to Weights and Biases."""
    wandb_project_name: str = "cleanRL"
    """Weights and Biases project."""
    wandb_entity: str = None
    """Weights and Biases entity."""
    capture_video: bool = False
    """Whether evaluation/training video should be captured."""
    save_model: bool = False
    """Whether to persist the learned model."""
    upload_model: bool = False
    """Whether to publish the learned model."""
    hf_entity: str = ""
    """Hugging Face user or organization."""

    env_id: str = "CartPole-v1"
    """Gymnasium environment identifier."""
    total_timesteps: int = 500000
    """Number of environment interactions."""
    learning_rate: float = 2.5e-4
    """Adam learning rate."""
    num_envs: int = 1
    """Number of simultaneous environments."""
    buffer_size: int = 10000
    """Replay memory capacity."""
    gamma: float = 0.99
    """Discount factor."""
    tau: float = 1.0
    """Target-network interpolation coefficient."""
    target_network_frequency: int = 500
    """Target network update interval."""
    batch_size: int = 128
    """Replay batch size."""
    start_e: float = 1
    """Initial exploration probability."""
    end_e: float = 0.05
    """Final exploration probability."""
    exploration_fraction: float = 0.5
    """Portion of training used for epsilon annealing."""
    learning_starts: int = 10000
    """Number of steps collected before learning."""
    train_frequency: int = 10
    """Optimization interval."""


def make_env(env_id, seed, idx, capture_video, run_name):
    def create_environment():
        if capture_video and idx == 0:
            created_env = gym.make(env_id, render_mode="rgb_array")
            created_env = gym.wrappers.RecordVideo(created_env, f"videos/{run_name}")
        else:
            created_env = gym.make(env_id)
        created_env = gym.wrappers.RecordEpisodeStatistics(created_env)
        created_env.action_space.seed(seed)
        return created_env

    return create_environment


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        feature_count = np.array(env.single_observation_space.shape).prod()
        action_count = env.single_action_space.n
        self.network = nn.Sequential(
            nn.Linear(feature_count, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_count),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    increment = (end_e - start_e) / duration
    return max(start_e + increment * t, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

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
        % "\n".join(f"|{name}|{value}|" for name, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + env_index, env_index, args.capture_video, run_name)
            for env_index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

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
                values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        replay_next_obs = next_obs.copy()
        for env_index, was_truncated in enumerate(truncations):
            if was_truncated:
                replay_next_obs[env_index] = infos["final_observation"][env_index]

        rb.add(obs, replay_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_q_values, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * next_q_values * (1 - data.dones.flatten())

                selected_q_values = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, selected_q_values)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", selected_q_values.mean().item(), global_step)
                    steps_per_second = int(global_step / (time.time() - start_time))
                    print("SPS:", steps_per_second)
                    writer.add_scalar("charts/SPS", steps_per_second, global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_parameter, online_parameter in zip(target_network.parameters(), q_network.parameters()):
                    target_parameter.data.copy_(
                        args.tau * online_parameter.data + (1.0 - args.tau) * target_parameter.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )

        for index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, index)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repository_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(
                args,
                episodic_returns,
                repository_name,
                "DQN",
                model_path,
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()