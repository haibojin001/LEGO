import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Sequence

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from huggingface_hub import hf_hub_download
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from cleanrl.dqn_atari_jax import QNetwork as TeacherModel
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.evals.dqn_jax_eval import evaluate


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""

    teacher_policy_hf_repo: str = None
    """the huggingface repo of the teacher policy"""
    teacher_model_exp_name: str = "dqn_atari_jax"
    """the experiment name of the teacher model"""
    teacher_eval_episodes: int = 10
    """the number of episodes to run the teacher policy evaluate"""
    teacher_steps: int = 500000
    """the number of steps to run the teacher policy to generate the replay buffer"""
    offline_steps: int = 500000
    """the number of steps to run the student policy with the teacher's replay buffer"""
    temperature: float = 1.0
    """the temperature parameter for qdagger"""


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


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        shortcut = x
        x = nn.relu(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3))(x)
        return shortcut + x


class ConvSequence(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.channels, kernel_size=(3, 3))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(self.channels)(x)
        return ResidualBlock(self.channels)(x)


class QNetwork(nn.Module):
    action_dim: int
    channelss: Sequence[int] = (16, 32, 32)

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        for width in self.channelss:
            x = ConvSequence(width)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(start_e + slope * t, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    if args.teacher_policy_hf_repo is None:
        args.teacher_policy_hf_repo = f"cleanrl/{args.env_id}-{args.teacher_model_exp_name}-seed1"

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
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(action_dim=envs.single_action_space.n, channelss=(16, 32, 32))
    initial_variables = q_network.init(model_key, envs.observation_space.sample())
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=initial_variables,
        target_params=initial_variables,
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    q_network.apply = jax.jit(q_network.apply)

    teacher_model_path = hf_hub_download(
        repo_id=args.teacher_policy_hf_repo,
        filename=f"{args.teacher_model_exp_name}.cleanrl_model",
    )
    teacher_model = TeacherModel(action_dim=envs.single_action_space.n)
    teacher_variables = teacher_model.init(jax.random.PRNGKey(args.seed), envs.observation_space.sample())
    with open(teacher_model_path, "rb") as file:
        teacher_params = flax.serialization.from_bytes(teacher_variables, file.read())
    teacher_model.apply = jax.jit(teacher_model.apply)

    teacher_returns = evaluate(
        teacher_model_path,
        make_env,
        args.env_id,
        eval_episodes=args.teacher_eval_episodes,
        run_name=f"{run_name}-teacher-eval",
        Model=TeacherModel,
        epsilon=args.end_e,
        capture_video=False,
    )
    writer.add_scalar("charts/teacher/avg_episodic_return", np.mean(teacher_returns), 0)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    observations, _ = envs.reset(seed=args.seed)
    for _ in track(range(args.teacher_steps), description="Collecting teacher transitions"):
        teacher_values = np.asarray(jax.device_get(teacher_model.apply(teacher_params, observations)))
        actions = teacher_values.argmax(axis=1)
        next_observations, rewards, terminations, truncations, infos = envs.step(actions)

        stored_next_observations = next_observations.copy()
        for env_index, truncated in enumerate(truncations):
            if truncated:
                stored_next_observations[env_index] = infos["final_observation"][env_index]

        rb.add(observations, stored_next_observations, actions, rewards, terminations, infos)
        observations = next_observations

    @jax.jit
    def update(state, batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_dones):
        def loss_function(params):
            online_q = q_network.apply(params, batch_observations)
            chosen_q = jnp.take_along_axis(online_q, batch_actions, axis=1).squeeze(axis=1)

            next_q = q_network.apply(state.target_params, batch_next_observations)
            bootstrap = jnp.max(next_q, axis=1)
            target = batch_rewards.squeeze(axis=1) + args.gamma * (1.0 - batch_dones.squeeze(axis=1)) * bootstrap
            td_loss = jnp.mean(optax.l2_loss(chosen_q, jax.lax.stop_gradient(target)))

            teacher_q = teacher_model.apply(teacher_params, batch_observations)
            teacher_probabilities = jax.nn.softmax(teacher_q / args.temperature, axis=1)
            distillation_loss = jnp.mean(
                optax.softmax_cross_entropy(online_q / args.temperature, teacher_probabilities)
            )
            return td_loss + distillation_loss, (td_loss, distillation_loss, chosen_q.mean())

        (loss, auxiliary), gradients = jax.value_and_grad(loss_function, has_aux=True)(state.params)
        state = state.apply_gradients(grads=gradients)
        return state, loss, auxiliary

    def sample_and_update(state):
        sampled = rb.sample(args.batch_size)
        batch_observations = jnp.asarray(np.asarray(sampled.observations))
        batch_actions = jnp.asarray(np.asarray(sampled.actions), dtype=jnp.int32)
        batch_rewards = jnp.asarray(np.asarray(sampled.rewards), dtype=jnp.float32)
        batch_next_observations = jnp.asarray(np.asarray(sampled.next_observations))
        batch_dones = jnp.asarray(np.asarray(sampled.dones), dtype=jnp.float32)
        return update(
            state,
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_next_observations,
            batch_dones,
        )

    for offline_step in track(range(args.offline_steps), description="Offline QDagger training"):
        q_state, loss, values = sample_and_update(q_state)
        if offline_step % args.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
            )
        if offline_step % 100 == 0:
            td_loss, distillation_loss, q_mean = values
            writer.add_scalar("losses/offline_loss", float(loss), offline_step)
            writer.add_scalar("losses/offline_td_loss", float(td_loss), offline_step)
            writer.add_scalar("losses/offline_qdagger_loss", float(distillation_loss), offline_step)
            writer.add_scalar("losses/offline_q_values", float(q_mean), offline_step)

    start_time = time.time()
    episodic_returns = deque(maxlen=100)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            int(args.exploration_fraction * args.total_timesteps),
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()])
        else:
            q_values = np.asarray(jax.device_get(q_network.apply(q_state.params, observations)))
            actions = q_values.argmax(axis=1)

        next_observations, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    episode_return = info["episode"]["r"]
                    episodic_returns.append(episode_return)
                    print(f"global_step={global_step}, episodic_return={episode_return}")
                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        stored_next_observations = next_observations.copy()
        for env_index, truncated in enumerate(truncations):
            if truncated:
                stored_next_observations[env_index] = infos["final_observation"][env_index]

        rb.add(observations, stored_next_observations, actions, rewards, terminations, infos)
        observations = next_observations

        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            q_state, loss, values = sample_and_update(q_state)
            td_loss, distillation_loss, q_mean = values
            writer.add_scalar("losses/td_loss", float(td_loss), global_step)
            writer.add_scalar("losses/qdagger_loss", float(distillation_loss), global_step)
            writer.add_scalar("losses/loss", float(loss), global_step)
            writer.add_scalar("losses/q_values", float(q_mean), global_step)

        if global_step > args.learning_starts and global_step % args.target_network_frequency == 0:
            q_state = q_state.replace(
                target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
            )

        if global_step % 1000 == 0:
            elapsed = max(time.time() - start_time, 1e-9)
            writer.add_scalar("charts/SPS", int(global_step / elapsed), global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)
            if episodic_returns:
                writer.add_scalar("charts/mean_episodic_return", np.mean(episodic_returns), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(q_state.params))

        evaluation_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-final-eval",
            Model=QNetwork,
            epsilon=args.end_e,
            capture_video=args.capture_video,
        )
        for episode_index, episode_return in enumerate(evaluation_returns):
            writer.add_scalar("eval/episodic_return", episode_return, episode_index)
        writer.add_scalar("eval/mean_episodic_return", np.mean(evaluation_returns), args.total_timesteps)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repository_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(
                args,
                evaluation_returns,
                repository_id,
                "Q-Dagger DQN",
                run_name,
            )

    envs.close()
    writer.close()