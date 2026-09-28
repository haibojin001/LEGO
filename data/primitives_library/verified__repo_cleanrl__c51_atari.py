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
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
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
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10
    buffer_size: int = 1000000
    gamma: float = 0.99
    target_network_frequency: int = 10000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4


def make_env(env_id, seed, idx, capture_video, run_name):
    def build_environment():
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

    return build_environment


class QNetwork(nn.Module):
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.n = env.single_action_space.n
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        output = self.network(x / 255.0)
        distributions = torch.softmax(output.view(len(x), self.n, self.n_atoms), dim=2)
        expected_values = (distributions * self.atoms).sum(dim=2)

        if action is None:
            action = expected_values.argmax(dim=1)

        return action, distributions[torch.arange(len(x), device=x.device), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    change_per_step = (end_e - start_e) / duration
    return max(start_e + change_per_step * t, end_e)


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
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(
        envs,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
    ).to(device)
    target_network = QNetwork(
        envs,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
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
            actions, _ = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        stored_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                stored_next_obs[index] = infos["final_observation"][index]

        rb.add(obs, stored_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    _, target_pmf = target_network.get_action(data.next_observations)

                    projected_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)
                    projected_atoms = projected_atoms.clamp(args.v_min, args.v_max)

                    atom_delta = target_network.atoms[1] - target_network.atoms[0]
                    positions = (projected_atoms - args.v_min) / atom_delta
                    lower = positions.floor().clamp(0, args.n_atoms - 1)
                    upper = positions.ceil().clamp(0, args.n_atoms - 1)

                    target_distribution = torch.zeros_like(target_pmf)
                    target_distribution.scatter_add_(
                        1,
                        lower.long(),
                        target_pmf * (upper.float() - positions + (lower == upper).float()),
                    )
                    target_distribution.scatter_add_(
                        1,
                        upper.long(),
                        target_pmf * (positions - lower.float()),
                    )

                _, current_pmf = q_network.get_action(data.observations, data.actions.flatten())
                loss = -(target_distribution * current_pmf.log()).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    current_q_values = (current_pmf * q_network.atoms).sum(dim=1)
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", current_q_values.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.c51_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
            capture_video=args.capture_video,
            model_kwargs={"n_atoms": args.n_atoms, "v_min": args.v_min, "v_max": args.v_max},
        )

        for index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, index)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repository_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(
                args,
                episodic_returns,
                repository_id,
                "C51",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()