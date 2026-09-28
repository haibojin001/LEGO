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
from torch.distributions.categorical import Categorical
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

    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""


def make_env(env_id, seed, idx, capture_video, run_name):
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
        environment.action_space.seed(seed)
        return environment

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_shape = envs.single_observation_space.shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            conv_output_size = self.conv(torch.zeros(1, *observation_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(conv_output_size, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        features = F.relu(self.conv(x / 255.0))
        features = F.relu(self.fc1(features))
        return self.fc_q(features)


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        observation_shape = envs.single_observation_space.shape

        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            conv_output_size = self.conv(torch.zeros(1, *observation_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(conv_output_size, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        features = F.relu(self.conv(x))
        features = F.relu(self.fc1(features))
        return self.fc_logits(features)

    def get_action(self, x):
        logits = self(x / 255.0)
        distribution = Categorical(logits=logits)
        sampled_action = distribution.sample()
        probabilities = distribution.probs
        log_probabilities = F.log_softmax(logits, dim=1)
        return sampled_action, log_probabilities, probabilities


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
        % ("\n".join(f"|{key}|{value}|" for key, value in vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr,
        eps=1e-4,
    )
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr, eps=1e-4)

    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(
            1 / torch.tensor(envs.single_action_space.n)
        )
        log_alpha = torch.zeros(1, device=device, requires_grad=True)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

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
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, "
                        f"episodic_return={info['episode']['r']}"
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

        real_next_obs = next_obs.copy()
        for idx, truncation in enumerate(truncations):
            if truncation:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    _, next_log_pi, next_action_probs = actor.get_action(
                        data.next_observations
                    )
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)

                    min_qf_next_target = (
                        next_action_probs
                        * (min_qf_next_target - alpha * next_log_pi)
                    ).sum(dim=1)

                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * min_qf_next_target

                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(
                    1, data.actions.long()
                ).view(-1)
                qf2_a_values = qf2_values.gather(
                    1, data.actions.long()
                ).view(-1)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                _, log_pi, action_probs = actor.get_action(data.observations)

                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)

                actor_loss = (
                    action_probs * (alpha * log_pi - min_qf_values)
                ).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    alpha_loss = (
                        action_probs.detach()
                        * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                    ).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                if global_step % 100 == 0:
                    writer.add_scalar(
                        "losses/qf1_values",
                        qf1_a_values.mean().item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "losses/qf2_values",
                        qf2_a_values.mean().item(),
                        global_step,
                    )
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                    writer.add_scalar(
                        "losses/actor_loss",
                        actor_loss.item(),
                        global_step,
                    )
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss",
                            alpha_loss.item(),
                            global_step,
                        )
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

            if global_step % args.target_network_frequency == 0:
                for target_param, param in zip(
                    qf1_target.parameters(), qf1.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for target_param, param in zip(
                    qf2_target.parameters(), qf2.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

    envs.close()
    writer.close()