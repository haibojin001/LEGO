import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tyro
from rich.pretty import pprint
from torch.distributions.categorical import Categorical
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

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    local_num_envs: int = 8
    """the number of parallel game environments (in the local rank)"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    device_ids: List[int] = field(default_factory=lambda: [])
    """the device ids that subprocess workers will use"""
    backend: Literal["gloo", "nccl", "mpi"] = "gloo"
    """the backend for distributed training"""

    local_batch_size: int = 0
    """the local batch size in the local rank (computed in runtime)"""
    local_minibatch_size: int = 0
    """the local mini-batch size in the local rank (computed in runtime)"""
    num_envs: int = 0
    """the number of parallel game environments (computed in runtime)"""
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    world_size: int = 0
    """the number of processes (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
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
        return environment

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        features = self.network(x / 255.0)
        distribution = Categorical(logits=self.actor(features))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(features)


if __name__ == "__main__":
    args = tyro.cli(Args)
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))

    args.local_batch_size = int(args.local_num_envs * args.num_steps)
    args.local_minibatch_size = int(args.local_batch_size // args.num_minibatches)
    args.num_envs = int(args.local_num_envs * args.world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    if args.world_size > 1:
        dist.init_process_group(args.backend, rank=local_rank, world_size=args.world_size)
    else:
        warnings.warn(
            """
Not using distributed mode!
If you want to use distributed mode, please execute this script with 'torchrun'.
E.g., `torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py`
            """
        )

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = None

    if local_rank == 0:
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
            "|param|value|\n|-|-|\n"
            + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
        )
        pprint(args)

    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.device_ids:
        assert len(args.device_ids) == args.world_size, (
            "you must specify the same number of device ids as `--nproc_per_node`"
        )
        if torch.cuda.is_available() and args.cuda:
            device = torch.device(f"cuda:{args.device_ids[local_rank]}")
        else:
            device = torch.device("cpu")
    else:
        available_devices = torch.cuda.device_count()
        if available_devices < args.world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, index, args.capture_video, run_name) for index in range(args.local_num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    torch.manual_seed(args.seed)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros(
        (args.num_steps, args.local_num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.local_num_envs), device=device)
    values = torch.zeros((args.num_steps, args.local_num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, device=device)
    next_done = torch.zeros(args.local_num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            fraction = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = fraction * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            stepped_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            next_done = torch.as_tensor(np.logical_or(terminated, truncated), device=device, dtype=torch.float32)
            rewards[step] = torch.as_tensor(reward, device=device, dtype=torch.float32)
            next_obs = torch.as_tensor(stepped_obs, device=device)

            if local_rank == 0 and "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        episode_return = info["episode"]["r"]
                        episode_length = info["episode"]["l"]
                        print(f"global_step={global_step}, episodic_return={episode_return}")
                        writer.add_scalar("charts/episodic_return", episode_return, global_step)
                        writer.add_scalar("charts/episodic_length", episode_length, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards)
            gae = 0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    nonterminal = 1.0 - next_done
                    following_value = next_value
                else:
                    nonterminal = 1.0 - dones[step + 1]
                    following_value = values[step + 1]

                temporal_difference = rewards[step] + args.gamma * following_value * nonterminal - values[step]
                gae = temporal_difference + args.gamma * args.gae_lambda * nonterminal * gae
                advantages[step] = gae

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        batch_indices = np.arange(args.local_batch_size)
        clipfracs = []
        approximate_kl = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        policy_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, args.local_batch_size, args.local_minibatch_size):
                end = start + args.local_minibatch_size
                minibatch_indices = batch_indices[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[minibatch_indices],
                    b_actions.long()[minibatch_indices],
                )
                logratio = newlogprob - b_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approximate_kl = (-logratio).mean()
                    approximate_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                minibatch_advantages = b_advantages[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                unclipped_policy_loss = -minibatch_advantages * ratio
                clipped_policy_loss = -minibatch_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                policy_loss = torch.maximum(unclipped_policy_loss, clipped_policy_loss).mean()

                newvalue = newvalue.flatten()
                if args.clip_vloss:
                    raw_value_loss = (newvalue - b_returns[minibatch_indices]).pow(2)
                    constrained_value = b_values[minibatch_indices] + torch.clamp(
                        newvalue - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    clipped_value_loss = (constrained_value - b_returns[minibatch_indices]).pow(2)
                    value_loss = 0.5 * torch.maximum(raw_value_loss, clipped_value_loss).mean()
                else:
                    value_loss = 0.5 * (newvalue - b_returns[minibatch_indices]).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

                optimizer.zero_grad()
                loss.backward()

                if args.world_size > 1:
                    for parameter in agent.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                            parameter.grad.div_(args.world_size)

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approximate_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        variance_y = np.var(y_true)
        explained_variance = np.nan if variance_y == 0 else 1.0 - np.var(y_true - y_pred) / variance_y

        if local_rank == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approximate_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approximate_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_variance, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if writer is not None:
        writer.close()
    if args.world_size > 1:
        dist.destroy_process_group()