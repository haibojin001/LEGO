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
from torch import distributions as td
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
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

    env_id: str = "starpilot"
    """the id of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    adv_norm_fullbatch: bool = True
    """Toggle full batch advantage normalization as used in PPG code"""
    clip_coef: float = 0.2
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

    n_iteration: int = 32
    """N_pi: the number of policy update in the policy phase """
    e_policy: int = 1
    """E_pi: the number of policy update in the policy phase """
    v_value: int = 1
    """E_V: the number of policy update in the policy phase """
    e_auxiliary: int = 6
    """E_aux:the K epochs to update the policy"""
    beta_clone: float = 1.0
    """the behavior cloning coefficient"""
    num_aux_rollouts: int = 4
    """the number of mini batch in the auxiliary phase"""
    n_aux_grad_accum: int = 1
    """the number of gradient accumulation in mini batch"""

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_phases: int = 0
    """the number of phases (computed in runtime)"""
    aux_batch_rollouts: int = 0
    """the number of rollouts in the auxiliary phase (computed in runtime)"""


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = torch.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert torch.equal(a, c)


class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        scale = np.sqrt(scale)
        self.conv0 = layer_init_normed(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_dim=(1, 2, 3),
            scale=scale,
        )
        self.conv1 = layer_init_normed(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            norm_dim=(1, 2, 3),
            scale=scale,
        )

    def forward(self, x):
        residual = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + residual


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = layer_init_normed(
            nn.Conv2d(input_shape[0], out_channels, kernel_size=3, padding=1),
            norm_dim=(1, 2, 3),
            scale=1.0,
        )
        block_scale = scale / np.sqrt(2)
        self.res_block0 = ResidualBlock(out_channels, block_scale)
        self.res_block1 = ResidualBlock(out_channels, block_scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _, height, width = self._input_shape
        return self._out_channels, (height + 1) // 2, (width + 1) // 2


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        height, width, channels = envs.single_observation_space.shape
        shape = (channels, height, width)
        modules = []
        channel_sizes = [16, 32, 32]
        sequence_scale = 1 / np.sqrt(len(channel_sizes))

        for channels_out in channel_sizes:
            sequence = ConvSequence(shape, channels_out, sequence_scale)
            modules.append(sequence)
            shape = sequence.get_output_shape()

        encoder_top = layer_init_normed(
            nn.Linear(shape[0] * shape[1] * shape[2], 256),
            norm_dim=1,
            scale=1.4,
        )
        modules.extend([nn.Flatten(), nn.ReLU(), encoder_top, nn.ReLU()])

        self.network = nn.Sequential(*modules)
        self.actor = layer_init_normed(
            nn.Linear(256, envs.single_action_space.n),
            norm_dim=1,
            scale=0.1,
        )
        self.critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        self.aux_critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        distribution = Categorical(logits=self.actor(hidden))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(hidden.detach())

    def get_value(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return self.critic(hidden)

    def get_pi_value_and_aux_value(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return Categorical(logits=self.actor(hidden)), self.critic(hidden.detach()), self.aux_critic(hidden)

    def get_pi(self, x):
        hidden = self.network(x.permute((0, 3, 1, 2)) / 255.0)
        return Categorical(logits=self.actor(hidden))


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.num_phases = int(args.num_iterations // args.n_iteration)
    args.aux_batch_rollouts = int(args.num_envs * args.n_iteration)

    assert args.v_value == 1, "Multiple value epoch (v_value != 1) is not supported yet"
    assert args.batch_size % args.num_minibatches == 0
    assert args.aux_batch_rollouts % args.num_aux_rollouts == 0
    assert args.num_aux_rollouts % args.n_aux_grad_accum == 0

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
        % ("\n".join(f"|{key}|{value}|" for key, value in vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.backends.cudnn.benchmark = not args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        num_levels=200,
        start_level=0,
        distribution_mode="easy",
    )
    envs = gym.wrappers.TransformObservation(envs, lambda observation: observation["rgb"])
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    aux_obs = torch.empty(
        (args.n_iteration, args.num_envs, args.num_steps) + observation_shape,
        dtype=torch.uint8,
        device="cpu",
    )
    aux_returns = torch.empty(
        (args.n_iteration, args.num_envs, args.num_steps),
        dtype=torch.float32,
        device="cpu",
    )

    reset_result = envs.reset()
    if isinstance(reset_result, tuple):
        reset_result = reset_result[0]
    next_obs = torch.tensor(reset_result, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    global_step = 0
    start_time = time.time()

    last_pg_loss = torch.tensor(0.0)
    last_v_loss = torch.tensor(0.0)
    last_entropy_loss = torch.tensor(0.0)
    last_old_approx_kl = torch.tensor(0.0)
    last_approx_kl = torch.tensor(0.0)
    last_clipfrac = 0.0
    last_aux_value_loss = torch.tensor(0.0)
    last_kl_loss = torch.tensor(0.0)

    for phase in range(1, args.num_phases + 1):
        for iteration in range(args.n_iteration):
            current_iteration = (phase - 1) * args.n_iteration + iteration + 1

            if args.anneal_lr:
                fraction = 1.0 - (current_iteration - 1.0) / args.num_iterations
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

                step_result = envs.step(action.cpu().numpy())
                if len(step_result) == 5:
                    next_observation, reward, terminated, truncated, infos = step_result
                    done = np.logical_or(terminated, truncated)
                else:
                    next_observation, reward, done, infos = step_result

                rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
                next_obs = torch.tensor(next_observation, dtype=torch.float32, device=device)
                next_done = torch.tensor(done, dtype=torch.float32, device=device)

                if isinstance(infos, dict):
                    infos_iterable = [infos]
                else:
                    infos_iterable = infos

                for info in infos_iterable:
                    if info is not None and "episode" in info:
                        episode = info["episode"]
                        episodic_return = episode["r"]
                        episodic_length = episode["l"]
                        if isinstance(episodic_return, np.ndarray):
                            episodic_return = episodic_return.item()
                        if isinstance(episodic_length, np.ndarray):
                            episodic_length = episodic_length.item()
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length, global_step)

            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = torch.zeros(args.num_envs, device=device)

                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        next_nonterminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_nonterminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]

                    delta = rewards[step] + args.gamma * next_values.flatten() * next_nonterminal - values[step]
                    advantages[step] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * next_nonterminal * lastgaelam
                    )

                returns = advantages + values

            aux_obs[iteration].copy_(obs.detach().transpose(0, 1).to(dtype=torch.uint8, device="cpu"))
            aux_returns[iteration].copy_(returns.detach().transpose(0, 1).to(device="cpu"))

            b_obs = obs.reshape((-1,) + observation_shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            if args.adv_norm_fullbatch:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            batch_indices = np.arange(args.batch_size)
            clipfracs = []
            stop_early = False

            for _ in range(args.e_policy):
                np.random.shuffle(batch_indices)

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[minibatch_indices],
                        b_actions.long()[minibatch_indices],
                    )
                    logratio = newlogprob - b_logprobs[minibatch_indices]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        last_old_approx_kl = (-logratio).mean()
                        last_approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                    minibatch_advantages = b_advantages[minibatch_indices]
                    if not args.adv_norm_fullbatch:
                        minibatch_advantages = (
                            minibatch_advantages - minibatch_advantages.mean()
                        ) / (minibatch_advantages.std() + 1e-8)

                    policy_loss_1 = -minibatch_advantages * ratio
                    policy_loss_2 = -minibatch_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    last_pg_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                    newvalue = newvalue.flatten()
                    if args.clip_vloss:
                        unclipped_value_loss = (newvalue - b_returns[minibatch_indices]) ** 2
                        clipped_value = b_values[minibatch_indices] + torch.clamp(
                            newvalue - b_values[minibatch_indices],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        clipped_value_loss = (clipped_value - b_returns[minibatch_indices]) ** 2
                        last_v_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
                    else:
                        last_v_loss = 0.5 * ((newvalue - b_returns[minibatch_indices]) ** 2).mean()

                    last_entropy_loss = entropy.mean()
                    loss = last_pg_loss - args.ent_coef * last_entropy_loss + args.vf_coef * last_v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    if args.target_kl is not None and last_approx_kl > args.target_kl:
                        stop_early = True
                        break

                if stop_early:
                    break

            last_clipfrac = float(np.mean(clipfracs)) if clipfracs else 0.0

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", last_v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", last_pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", last_entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", last_old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", last_approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", last_clipfrac, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        flattened_aux_obs = aux_obs.reshape(
            (args.aux_batch_rollouts, args.num_steps) + observation_shape
        )
        flattened_aux_returns = aux_returns.reshape(args.aux_batch_rollouts, args.num_steps)

        with torch.no_grad():
            old_logits = torch.empty(
                (args.aux_batch_rollouts, args.num_steps, envs.single_action_space.n),
                dtype=torch.float32,
                device="cpu",
            )
            rollout_chunk = max(1, args.aux_batch_rollouts // args.num_aux_rollouts)
            for start in range(0, args.aux_batch_rollouts, rollout_chunk):
                end = min(start + rollout_chunk, args.aux_batch_rollouts)
                batch_observations = flatten01(flattened_aux_obs[start:end]).to(
                    device=device,
                    dtype=torch.float32,
                )
                logits = agent.get_pi(batch_observations).logits
                old_logits[start:end].copy_(
                    unflatten01(logits, (end - start, args.num_steps)).to(device="cpu")
                )

        auxiliary_indices = np.arange(args.aux_batch_rollouts)
        aux_rollout_batch_size = args.aux_batch_rollouts // args.num_aux_rollouts

        for _ in range(args.e_auxiliary):
            np.random.shuffle(auxiliary_indices)
            optimizer.zero_grad()

            for auxiliary_step, start in enumerate(
                range(0, args.aux_batch_rollouts, aux_rollout_batch_size)
            ):
                end = start + aux_rollout_batch_size
                minibatch_indices = auxiliary_indices[start:end]

                minibatch_obs = flatten01(flattened_aux_obs[minibatch_indices]).to(
                    device=device,
                    dtype=torch.float32,
                )
                minibatch_returns = flatten01(
                    flattened_aux_returns[minibatch_indices]
                ).to(device=device)
                minibatch_logits = flatten01(old_logits[minibatch_indices]).to(device=device)

                old_pi = Categorical(logits=minibatch_logits)
                new_pi, _, new_aux_value = agent.get_pi_value_and_aux_value(minibatch_obs)

                last_aux_value_loss = 0.5 * ((new_aux_value.flatten() - minibatch_returns) ** 2).mean()
                last_kl_loss = td.kl_divergence(old_pi, new_pi).mean()
                auxiliary_loss = last_aux_value_loss + args.beta_clone * last_kl_loss
                (auxiliary_loss / args.n_aux_grad_accum).backward()

                if (
                    (auxiliary_step + 1) % args.n_aux_grad_accum == 0
                    or auxiliary_step + 1 == args.num_aux_rollouts
                ):
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

        writer.add_scalar("losses/auxiliary_value_loss", last_aux_value_loss.item(), global_step)
        writer.add_scalar("losses/behavior_clone_loss", last_kl_loss.item(), global_step)
        print(f"SPS: {int(global_step / (time.time() - start_time))}")

    envs.close()
    writer.close()