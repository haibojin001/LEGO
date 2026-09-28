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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of the experiment"""
    seed: int = 1
    """the seed used for reproducibility"""
    torch_deterministic: bool = True
    """whether cuDNN uses deterministic algorithms"""
    cuda: bool = True
    """whether CUDA should be used when available"""
    track: bool = False
    """whether to log the run to Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the Weights and Biases project"""
    wandb_entity: str = None
    """the Weights and Biases entity"""
    capture_video: bool = False
    """whether to record environment videos"""
    save_model: bool = False
    """whether to store the trained parameters"""
    upload_model: bool = False
    """whether to upload the stored model to Hugging Face"""
    hf_entity: str = ""
    """the Hugging Face organization or username"""

    env_id: str = "HalfCheetah-v4"
    """the environment identifier"""
    total_timesteps: int = 1000000
    """total environment interactions"""
    learning_rate: float = 3e-4
    """Adam learning rate"""
    num_envs: int = 1
    """number of vectorized environments"""
    num_steps: int = 2048
    """steps collected per environment before every update"""
    anneal_lr: bool = True
    """whether learning rate is linearly annealed"""
    gamma: float = 0.99
    """reward discount"""
    gae_lambda: float = 0.95
    """GAE trace parameter"""
    num_minibatches: int = 32
    """minibatches per PPO epoch"""
    update_epochs: int = 10
    """number of PPO optimization epochs"""
    norm_adv: bool = True
    """whether advantages are standardized"""
    clip_coef: float = 0.2
    """PPO clipping range"""
    clip_vloss: bool = True
    """whether value targets use clipping"""
    ent_coef: float = 0.0
    """entropy-loss weight"""
    vf_coef: float = 0.5
    """value-loss weight"""
    max_grad_norm: float = 0.5
    """gradient norm cap"""
    target_kl: float = None
    """optional early-stop KL threshold"""

    batch_size: int = 0
    """rollout batch size, assigned at runtime"""
    minibatch_size: int = 0
    """minibatch size, assigned at runtime"""
    num_iterations: int = 0
    """number of optimization iterations, assigned at runtime"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def build_environment():
        if capture_video and idx == 0:
            environment = gym.make(env_id, render_mode="rgb_array")
            environment = gym.wrappers.RecordVideo(environment, f"videos/{run_name}")
        else:
            environment = gym.make(env_id)

        environment = gym.wrappers.FlattenObservation(environment)
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment = gym.wrappers.ClipAction(environment)
        environment = gym.wrappers.NormalizeObservation(environment)
        environment = gym.wrappers.TransformObservation(environment, lambda value: np.clip(value, -10, 10))
        environment = gym.wrappers.NormalizeReward(environment, gamma=gamma)
        environment = gym.wrappers.TransformReward(environment, lambda value: np.clip(value, -10, 10))
        return environment

    return build_environment


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_features = int(np.array(envs.single_observation_space.shape).prod())
        action_features = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_features, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_features), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_features))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        log_std = self.actor_logstd.expand_as(mean)
        distribution = Normal(mean, torch.exp(log_std))
        if action is None:
            action = distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            self.critic(x),
        )


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
        % "\n".join(f"|{name}|{value}|" for name, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, index, args.capture_video, run_name, args.gamma)
            for index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    initial_observation, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(initial_observation).to(device)
    next_done = torch.zeros(args.num_envs, device=device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            progress = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = progress * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, action_logprob, _, predicted_value = agent.get_action_and_value(next_obs)
                values[step] = predicted_value.flatten()

            actions[step] = action
            logprobs[step] = action_logprob

            observation, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            finished = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(observation).to(device)
            next_done = torch.Tensor(finished).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = info["episode"]["r"]
                        episodic_length = info["episode"]["l"]
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", episodic_length, global_step)

        with torch.no_grad():
            bootstrap_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            running_advantage = 0

            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    continuation = 1.0 - next_done
                    following_value = bootstrap_value
                else:
                    continuation = 1.0 - dones[step + 1]
                    following_value = values[step + 1]

                td_error = rewards[step] + args.gamma * following_value * continuation - values[step]
                running_advantage = td_error + args.gamma * args.gae_lambda * continuation * running_advantage
                advantages[step] = running_advantage

            returns = advantages + values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        batch_indices = np.arange(args.batch_size)
        clipfracs = []

        for _ in range(args.update_epochs):
            np.random.shuffle(batch_indices)

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_indices = batch_indices[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[minibatch_indices], b_actions[minibatch_indices]
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

                unclipped_objective = -minibatch_advantages * ratio
                clipped_objective = -minibatch_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(unclipped_objective, clipped_objective).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    unclipped_value_loss = (newvalue - b_returns[minibatch_indices]) ** 2
                    clipped_values = b_values[minibatch_indices] + torch.clamp(
                        newvalue - b_values[minibatch_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    clipped_value_loss = (clipped_values - b_returns[minibatch_indices]) ** 2
                    v_loss = 0.5 * torch.max(unclipped_value_loss, clipped_value_loss).mean()
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
        variance = np.var(y_true)
        explained_variance = np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )

        for index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, index)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repository = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            if args.hf_entity:
                repository = f"{args.hf_entity}/{repository}"

            push_to_hub(
                args,
                episodic_returns,
                repository,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()