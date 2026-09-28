import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, makes cuDNN deterministic"""
    cuda: bool = True
    """if toggled, CUDA will be used when available"""
    track: bool = False
    """if toggled, use Weights and Biases tracking"""
    wandb_project_name: str = "cleanRL"
    """Weights and Biases project name"""
    wandb_entity: str = None
    """Weights and Biases entity"""
    capture_video: bool = False
    """whether to record videos"""

    env_id: str = "MontezumaRevenge-v5"
    """environment identifier"""
    total_timesteps: int = 2000000000
    """total environment transitions"""
    learning_rate: float = 1e-4
    """optimizer learning rate"""
    num_envs: int = 128
    """number of environments"""
    num_steps: int = 128
    """rollout horizon"""
    anneal_lr: bool = True
    """whether to linearly anneal learning rate"""
    gamma: float = 0.999
    """extrinsic reward discount"""
    gae_lambda: float = 0.95
    """GAE lambda"""
    num_minibatches: int = 4
    """number of minibatches per epoch"""
    update_epochs: int = 4
    """number of optimization epochs"""
    norm_adv: bool = True
    """whether to standardize advantages"""
    clip_coef: float = 0.1
    """PPO clipping range"""
    clip_vloss: bool = True
    """whether to clip value losses"""
    ent_coef: float = 0.001
    """entropy loss coefficient"""
    vf_coef: float = 0.5
    """value loss coefficient"""
    max_grad_norm: float = 0.5
    """gradient clipping threshold"""
    target_kl: float = None
    """optional approximate KL cutoff"""

    update_proportion: float = 0.25
    """fraction of samples used for RND predictor updates"""
    int_coef: float = 1.0
    """intrinsic reward coefficient"""
    ext_coef: float = 2.0
    """extrinsic reward coefficient"""
    int_gamma: float = 0.99
    """intrinsic reward discount"""
    num_iterations_obs_norm_init: int = 50
    """iterations used to initialize observation normalization"""

    batch_size: int = 0
    """runtime-computed rollout batch size"""
    minibatch_size: int = 0
    """runtime-computed minibatch size"""
    num_iterations: int = 0
    """runtime-computed number of updates"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return result

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[...] = self.episode_returns
        self.returned_episode_lengths[...] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        distribution = Categorical(logits=self.actor(hidden))
        residual = hidden + self.extra_layer(hidden)
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.critic_ext(residual),
            self.critic_int(residual),
        )

    def get_value(self, x):
        hidden = self.network(x / 255.0)
        residual = hidden + self.extra_layer(hidden)
        return self.critic_ext(residual), self.critic_int(residual)


class RNDModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        flattened_size = 64 * 7 * 7

        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(flattened_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(flattened_size, 512)),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def forward(self, next_obs):
        return self.predictor(next_obs), self.target(next_obs)


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.gamma * self.rewems + rews
        return self.rewems


def _normalized_rnd_observation(observations, rms, device):
    frames = observations[:, 3:4].detach().cpu().numpy()
    normalized = (frames - rms.mean) / np.sqrt(rms.var)
    normalized = np.clip(normalized, -5.0, 5.0)
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)


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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
    )
    envs = RecordEpisodeStatistics(envs)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action spaces are supported"

    agent = Agent(envs).to(device)
    rnd_model = RNDModel(envs.single_observation_space.shape, 512).to(device)
    optimizer = optim.Adam(
        list(agent.parameters()) + list(rnd_model.predictor.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    obs_rms = RunningMeanStd(shape=(1, 84, 84))
    reward_rms = RunningMeanStd()
    reward_filter = RewardForwardFilter(args.int_gamma)

    initial_obs = envs.reset()
    for _ in range(args.num_iterations_obs_norm_init * args.num_steps):
        sampled_actions = np.asarray([envs.single_action_space.sample() for _ in range(args.num_envs)])
        initial_obs, _, _, _ = envs.step(sampled_actions)
        obs_rms.update(initial_obs[:, 3:4])

    obs_shape = envs.single_observation_space.shape
    observations = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    next_observations = torch.zeros_like(observations)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    ext_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    int_rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    ext_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    int_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            progress = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = progress * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            observations[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(next_obs)

            actions[step] = action
            logprobs[step] = logprob
            ext_values[step] = value_ext.flatten()
            int_values[step] = value_int.flatten()

            stepped_obs, stepped_rewards, stepped_dones, infos = envs.step(action.cpu().numpy())
            next_obs = torch.as_tensor(stepped_obs, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(stepped_dones, dtype=torch.float32, device=device)
            next_observations[step] = next_obs
            ext_rewards[step] = torch.as_tensor(stepped_rewards, dtype=torch.float32, device=device)

            with torch.no_grad():
                rnd_input = _normalized_rnd_observation(next_obs, obs_rms, device)
                predicted, target = rnd_model(rnd_input)
                int_rewards[step] = 0.5 * (predicted - target).pow(2).mean(dim=1)

            if "r" in infos:
                terminals = infos.get("terminated", stepped_dones)
                for index, terminal in enumerate(terminals):
                    if terminal:
                        print(f"global_step={global_step}, episodic_return={infos['r'][index]}")
                        writer.add_scalar("charts/episodic_return", infos["r"][index], global_step)
                        writer.add_scalar("charts/episodic_length", infos["l"][index], global_step)

        discounted_intrinsic = []
        for reward in int_rewards.detach().cpu().numpy():
            discounted_intrinsic.append(reward_filter.update(reward))
        reward_rms.update(np.asarray(discounted_intrinsic))
        int_rewards /= float(np.sqrt(reward_rms.var))

        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext = next_value_ext.flatten()
            next_value_int = next_value_int.flatten()

            ext_advantages = torch.zeros_like(ext_rewards)
            int_advantages = torch.zeros_like(int_rewards)
            last_ext_gae = torch.zeros(args.num_envs, device=device)
            last_int_gae = torch.zeros(args.num_envs, device=device)

            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    nonterminal = 1.0 - next_done
                    following_ext_value = next_value_ext
                    following_int_value = next_value_int
                else:
                    nonterminal = 1.0 - dones[step + 1]
                    following_ext_value = ext_values[step + 1]
                    following_int_value = int_values[step + 1]

                ext_delta = ext_rewards[step] + args.gamma * following_ext_value * nonterminal - ext_values[step]
                int_delta = int_rewards[step] + args.int_gamma * following_int_value * nonterminal - int_values[step]
                last_ext_gae = ext_delta + args.gamma * args.gae_lambda * nonterminal * last_ext_gae
                last_int_gae = int_delta + args.int_gamma * args.gae_lambda * nonterminal * last_int_gae
                ext_advantages[step] = last_ext_gae
                int_advantages[step] = last_int_gae

            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        batch_obs = observations.reshape((-1,) + obs_shape)
        batch_next_obs = next_observations.reshape((-1,) + obs_shape)
        batch_actions = actions.reshape(-1)
        batch_logprobs = logprobs.reshape(-1)
        batch_ext_advantages = ext_advantages.reshape(-1)
        batch_int_advantages = int_advantages.reshape(-1)
        batch_ext_returns = ext_returns.reshape(-1)
        batch_int_returns = int_returns.reshape(-1)
        batch_ext_values = ext_values.reshape(-1)
        batch_int_values = int_values.reshape(-1)

        obs_rms.update(batch_next_obs[:, 3:4].detach().cpu().numpy())

        batch_indices = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0, device=device)
        old_approx_kl = torch.tensor(0.0, device=device)
        pg_loss = torch.tensor(0.0, device=device)
        v_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)
        forward_loss = torch.tensor(0.0, device=device)

        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_indices = batch_indices[start : start + args.minibatch_size]

                _, newlogprob, entropy, new_ext_value, new_int_value = agent.get_action_and_value(
                    batch_obs[mb_indices], batch_actions[mb_indices]
                )
                logratio = newlogprob - batch_logprobs[mb_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_ext_adv = batch_ext_advantages[mb_indices]
                mb_int_adv = batch_int_advantages[mb_indices]
                combined_advantages = args.ext_coef * mb_ext_adv + args.int_coef * mb_int_adv
                if args.norm_adv:
                    combined_advantages = (combined_advantages - combined_advantages.mean()) / (
                        combined_advantages.std() + 1e-8
                    )

                unclipped_policy_loss = -combined_advantages * ratio
                clipped_policy_loss = -combined_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.maximum(unclipped_policy_loss, clipped_policy_loss).mean()

                new_ext_value = new_ext_value.flatten()
                new_int_value = new_int_value.flatten()

                if args.clip_vloss:
                    ext_unclipped = (new_ext_value - batch_ext_returns[mb_indices]).pow(2)
                    int_unclipped = (new_int_value - batch_int_returns[mb_indices]).pow(2)
                    ext_clipped_value = batch_ext_values[mb_indices] + torch.clamp(
                        new_ext_value - batch_ext_values[mb_indices], -args.clip_coef, args.clip_coef
                    )
                    int_clipped_value = batch_int_values[mb_indices] + torch.clamp(
                        new_int_value - batch_int_values[mb_indices], -args.clip_coef, args.clip_coef
                    )
                    ext_clipped = (ext_clipped_value - batch_ext_returns[mb_indices]).pow(2)
                    int_clipped = (int_clipped_value - batch_int_returns[mb_indices]).pow(2)
                    ext_value_loss = 0.5 * torch.maximum(ext_unclipped, ext_clipped).mean()
                    int_value_loss = 0.5 * torch.maximum(int_unclipped, int_clipped).mean()
                else:
                    ext_value_loss = 0.5 * (new_ext_value - batch_ext_returns[mb_indices]).pow(2).mean()
                    int_value_loss = 0.5 * (new_int_value - batch_int_returns[mb_indices]).pow(2).mean()

                v_loss = args.ext_coef * ext_value_loss + args.int_coef * int_value_loss
                entropy_loss = entropy.mean()

                normalized_next = _normalized_rnd_observation(batch_next_obs[mb_indices], obs_rms, device)
                predicted_feature, target_feature = rnd_model(normalized_next)
                prediction_errors = (predicted_feature - target_feature.detach()).pow(2).mean(dim=1)
                predictor_mask = (torch.rand(prediction_errors.shape, device=device) < args.update_proportion).float()
                forward_loss = (prediction_errors * predictor_mask).sum() / torch.clamp(predictor_mask.sum(), min=1.0)

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + forward_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(agent.parameters()) + list(rnd_model.predictor.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred = batch_ext_values.detach().cpu().numpy()
        y_true = batch_ext_returns.detach().cpu().numpy()
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1.0 - np.var(y_true - y_pred) / variance

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        writer.add_scalar("losses/forward_loss", forward_loss.item(), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()