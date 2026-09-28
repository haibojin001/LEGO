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
    """the seed used for reproducibility"""
    torch_deterministic: bool = True
    """whether to make cuDNN deterministic"""
    cuda: bool = True
    """whether to use CUDA when it is available"""
    track: bool = False
    """whether to log the run to Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """Weights and Biases project name"""
    wandb_entity: str = None
    """Weights and Biases entity"""
    capture_video: bool = False
    """whether to record environment videos"""
    save_model: bool = False
    """whether to save the trained actor"""
    upload_model: bool = False
    """whether to upload the saved model to Hugging Face"""
    hf_entity: str = ""
    """Hugging Face account or organization"""

    env_id: str = "Hopper-v4"
    """the Gymnasium environment identifier"""
    total_timesteps: int = 1000000
    """number of environment transitions"""
    learning_rate: float = 3e-4
    """optimizer learning rate"""
    buffer_size: int = int(1e6)
    """maximum replay buffer capacity"""
    gamma: float = 0.99
    """discount factor"""
    tau: float = 0.005
    """Polyak target-update coefficient"""
    batch_size: int = 256
    """minibatch size"""
    exploration_noise: float = 0.1
    """Gaussian exploration-noise scale"""
    learning_starts: int = 25e3
    """number of steps before updates begin"""
    policy_frequency: int = 2
    """number of critic updates per actor update"""


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
        features = torch.cat((x, a), dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc3(features)


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_size = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(observation_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_size)
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        normalized_action = torch.tanh(self.fc_mu(features))
        return normalized_action * self.action_scale + self.action_bias


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
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)

    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())

    q_optimizer = optim.Adam(qf1.parameters(), lr=args.learning_rate)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
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
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(
                    0,
                    actor.action_scale * args.exploration_noise,
                )
                actions = actions.cpu().numpy().clip(
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                writer.add_scalar(
                    "charts/episodic_return",
                    info["episode"]["r"],
                    global_step,
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    info["episode"]["l"],
                    global_step,
                )
                break

        real_next_obs = next_obs.copy()
        for idx, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_actions = target_actor(data.next_observations)
                next_q = qf1_target(data.next_observations, next_actions)
                target_q = (
                    data.rewards.flatten()
                    + (1 - data.dones.flatten()) * args.gamma * next_q.view(-1)
                )

            current_q = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(current_q, target_q)

            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(
                    data.observations,
                    actor(data.observations),
                ).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for parameter, target_parameter in zip(
                    actor.parameters(),
                    target_actor.parameters(),
                ):
                    target_parameter.data.copy_(
                        args.tau * parameter.data
                        + (1 - args.tau) * target_parameter.data
                    )

                for parameter, target_parameter in zip(
                    qf1.parameters(),
                    qf1_target.parameters(),
                ):
                    target_parameter.data.copy_(
                        args.tau * parameter.data
                        + (1 - args.tau) * target_parameter.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values",
                    current_q.mean().item(),
                    global_step,
                )
                writer.add_scalar(
                    "losses/qf1_loss",
                    qf1_loss.item(),
                    global_step,
                )
                writer.add_scalar(
                    "losses/actor_loss",
                    actor_loss.item(),
                    global_step,
                )
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                print(
                    "SPS:",
                    int(global_step / (time.time() - start_time)),
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(actor.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.ddpg_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Actor,
            device=device,
            exploration_noise=args.exploration_noise,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            if args.hf_entity:
                repo_id = f"{args.hf_entity}/{repo_id}"

            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DDPG",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()