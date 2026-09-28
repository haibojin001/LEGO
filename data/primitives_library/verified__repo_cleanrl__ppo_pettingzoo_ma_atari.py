import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--torch-deterministic",
        type=lambda value: bool(strtobool(value)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda value: bool(strtobool(value)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
    )
    parser.add_argument(
        "--track",
        type=lambda value: bool(strtobool(value)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL", help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument(
        "--capture_video",
        type=lambda value: bool(strtobool(value)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)",
    )

    parser.add_argument("--env-id", type=str, default="pong_v3", help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=20000000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16, help="the number of parallel game environments")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda value: bool(strtobool(value)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument(
        "--norm-adv",
        type=lambda value: bool(strtobool(value)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument(
        "--clip-vloss",
        type=lambda value: bool(strtobool(value)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        x = x.clone()
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.critic(hidden)


if __name__ == "__main__":
    args = parse_args()
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

    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gym")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    if args.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

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
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            fraction = 1.0 - (update - 1.0) / num_updates
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

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item:
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[step + 1]
                    nextvalues = values[step + 1]
                delta = rewards[step] + args.gamma * nextvalues * nextnonterminal - values[step]
                advantages[step] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
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
                    b_obs[minibatch_indices],
                    b_actions.long()[minibatch_indices],
                )
                logratio = newlogprob - b_logprobs[minibatch_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                minibatch_advantages = b_advantages[minibatch_indices]
                if args.norm_adv:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                policy_loss_unclipped = -minibatch_advantages * ratio
                policy_loss_clipped = -minibatch_advantages * torch.clamp(
                    ratio,
                    1 - args.clip_coef,
                    1 + args.clip_coef,
                )
                pg_loss = torch.max(policy_loss_unclipped, policy_loss_clipped).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    value_loss_unclipped = (newvalue - b_returns[minibatch_indices]) ** 2
                    clipped_value = b_values[minibatch_indices] + torch.clamp(
                        newvalue - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_loss_clipped = (clipped_value - b_returns[minibatch_indices]) ** 2
                    v_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
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

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        variance_y = np.var(y_true)
        explained_var = np.nan if variance_y == 0 else 1 - np.var(y_true - y_pred) / variance_y

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