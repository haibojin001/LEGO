import os
import random
import time
from dataclasses import dataclass

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "starpilot"
    total_timesteps: int = int(25e6)
    learning_rate: float = 5e-4
    num_envs: int = 64
    num_steps: int = 256
    anneal_lr: bool = False
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 3
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
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

    def get_output_shape(self):
        _, height, width = self._input_shape
        return self._out_channels, (height + 1) // 2, (width + 1) // 2

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        height, width, channels = envs.single_observation_space.shape
        current_shape = (channels, height, width)
        modules = []

        for channels_out in (16, 32, 32):
            sequence = ConvSequence(current_shape, channels_out)
            current_shape = sequence.get_output_shape()
            modules.append(sequence)

        modules.extend(
            [
                nn.Flatten(),
                nn.ReLU(),
                nn.Linear(current_shape[0] * current_shape[1] * current_shape[2], 256),
                nn.ReLU(),
            ]
        )
        self.network = nn.Sequential(*modules)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        x = x.permute((0, 3, 1, 2)) / 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        x = x.permute((0, 3, 1, 2)) / 255.0
        hidden = self.network(x)
        distribution = Categorical(logits=self.actor(hidden))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(hidden)


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

    envs = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
    )
    envs = gym.wrappers.TransformObservation(envs, lambda observation: observation["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

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

            next_obs_np, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item:
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

        with torch.no_grad():
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = agent.get_value(next_obs).reshape(1, -1)
                else:
                    nextnonterminal = 1.0 - dones[step + 1]
                    nextvalues = values[step + 1]
                delta = rewards[step] + args.gamma * nextvalues * nextnonterminal - values[step]
                advantages[step] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        batch_indices = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[minibatch_indices], b_actions.long()[minibatch_indices]
                )
                logratio = newlogprob - b_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                minibatch_advantages = b_advantages[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) / (
                        minibatch_advantages.std() + 1e-8
                    )

                policy_loss_1 = -minibatch_advantages * ratio
                policy_loss_2 = -minibatch_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    unclipped_v_loss = (newvalue - b_returns[minibatch_indices]) ** 2
                    clipped_value = b_values[minibatch_indices] + torch.clamp(
                        newvalue - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    clipped_v_loss = (clipped_value - b_returns[minibatch_indices]) ** 2
                    v_loss = 0.5 * torch.max(unclipped_v_loss, clipped_v_loss).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()