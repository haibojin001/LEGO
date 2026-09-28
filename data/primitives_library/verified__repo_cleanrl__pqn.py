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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed used for reproducibility"""
    torch_deterministic: bool = True
    """whether cudnn should operate deterministically"""
    cuda: bool = True
    """whether CUDA should be used when available"""
    track: bool = False
    """whether to send experiment data to Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """Weights and Biases project name"""
    wandb_entity: str = None
    """Weights and Biases entity"""
    capture_video: bool = False
    """whether evaluation video should be captured"""

    env_id: str = "CartPole-v1"
    """environment identifier"""
    total_timesteps: int = 500000
    """number of environment interactions"""
    learning_rate: float = 2.5e-4
    """optimizer learning rate"""
    num_envs: int = 4
    """number of vectorized environments"""
    num_steps: int = 128
    """rollout length per environment"""
    num_minibatches: int = 4
    """number of minibatches in an update"""
    update_epochs: int = 4
    """number of optimization epochs"""
    anneal_lr: bool = True
    """whether learning rate decays linearly"""
    gamma: float = 0.99
    """discount factor"""
    start_e: float = 1
    """initial exploration epsilon"""
    end_e: float = 0.05
    """final exploration epsilon"""
    exploration_fraction: float = 0.5
    """fraction of training used for epsilon decay"""
    max_grad_norm: float = 10.0
    """gradient clipping norm"""
    q_lambda: float = 0.65
    """lambda parameter for Q(lambda) returns"""


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        observation_size = np.array(env.single_observation_space.shape).prod()
        action_count = env.single_action_space.n

        self.network = nn.Sequential(
            layer_init(nn.Linear(observation_size, 120)),
            nn.LayerNorm(120),
            nn.ReLU(),
            layer_init(nn.Linear(120, 84)),
            nn.LayerNorm(84),
            nn.ReLU(),
            layer_init(nn.Linear(84, action_count)),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    rate = (end_e - start_e) / duration
    return max(start_e + rate * t, end_e)


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
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.RAdam(q_network.parameters(), lr=args.learning_rate)

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
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

            epsilon = linear_schedule(
                args.start_e,
                args.end_e,
                args.exploration_fraction * args.total_timesteps,
                global_step,
            )

            random_actions = torch.randint(
                0, envs.single_action_space.n, (args.num_envs,)
            ).to(device)

            with torch.no_grad():
                q_values = q_network(next_obs)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()

            use_random_action = torch.rand((args.num_envs,)).to(device) < epsilon
            action = torch.where(use_random_action, random_actions, max_actions)
            actions[step] = action

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
            returns = torch.zeros_like(rewards).to(device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(q_network(next_obs), dim=-1)
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = rewards[t] + args.gamma * (
                        args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value
                    ) * nextnonterminal

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(args.batch_size)

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predicted_values = q_network(b_obs[mb_inds]).gather(
                    1, b_actions.long()[mb_inds]
                ).squeeze()
                loss = 0.5 * F.mse_loss(b_returns[mb_inds], predicted_values)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))

    envs.close()
    writer.close()