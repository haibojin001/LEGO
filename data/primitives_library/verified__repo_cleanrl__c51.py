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

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, cudnn deterministic behavior is enabled"""
    cuda: bool = True
    """if toggled, cuda is used when available"""
    track: bool = False
    """if toggled, track the experiment with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the Weights and Biases project name"""
    wandb_entity: str = None
    """the Weights and Biases entity"""
    capture_video: bool = False
    """whether to record agent videos"""
    save_model: bool = False
    """whether to save the trained model"""
    upload_model: bool = False
    """whether to upload the trained model to Hugging Face"""
    hf_entity: str = ""
    """the Hugging Face user or organization name"""

    env_id: str = "CartPole-v1"
    """the environment id"""
    total_timesteps: int = 500000
    """the total number of environment steps"""
    learning_rate: float = 2.5e-4
    """the optimizer learning rate"""
    num_envs: int = 1
    """the number of environments"""
    n_atoms: int = 101
    """the number of categorical support atoms"""
    v_min: float = -100
    """the minimum support value"""
    v_max: float = 100
    """the maximum support value"""
    buffer_size: int = 10000
    """the replay buffer capacity"""
    gamma: float = 0.99
    """the discount factor"""
    target_network_frequency: int = 500
    """the target network update interval"""
    batch_size: int = 128
    """the sampled replay batch size"""
    start_e: float = 1
    """the initial exploration probability"""
    end_e: float = 0.05
    """the final exploration probability"""
    exploration_fraction: float = 0.5
    """the fraction of training over which epsilon is annealed"""
    learning_starts: int = 10000
    """the number of steps before training begins"""
    train_frequency: int = 10
    """the gradient update interval"""


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
    def __init__(self, env, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.env = env
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, n_atoms))
        self.n = env.single_action_space.n
        observation_size = np.array(env.single_observation_space.shape).prod()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x)
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(dim=2)
        if action is None:
            action = torch.argmax(q_values, dim=1)
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(start_e + slope * t, end_e)


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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
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

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)

    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
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
            actions, _ = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        for idx, truncation in enumerate(truncations):
            if truncation:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(data.next_observations)
                    next_atoms = data.rewards + args.gamma * target_network.atoms * (1 - data.dones)

                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)
                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)

                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs

                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(data.observations, data.actions.flatten())
                loss = -(target_pmfs * old_pmfs.log()).sum(dim=1).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_pmfs.detach().mean(), global_step)
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
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            push_to_hub(
                args,
                episodic_returns,
                repo_id=f"{args.env_id}-{args.exp_name}-seed{args.seed}",
                algo_name=args.exp_name,
                run_name=run_name,
            )

    envs.close()
    writer.close()