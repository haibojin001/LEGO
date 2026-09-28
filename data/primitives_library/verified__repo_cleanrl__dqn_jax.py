import os
import random
import time
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState as FlaxTrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-3]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


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
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        hidden = nn.Dense(120)(x)
        hidden = nn.relu(hidden)
        hidden = nn.Dense(84)(hidden)
        hidden = nn.relu(hidden)
        return nn.Dense(self.action_dim)(hidden)


class TrainState(FlaxTrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    gradient = (end_e - start_e) / duration
    return max(start_e + gradient * t, end_e)


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
        "|param|value|\n|-|-|\n%s"
        % "\n".join(f"|{name}|{value}|" for name, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, initialization_key = jax.random.split(key, 2)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + index, index, args.capture_video, run_name)
            for index in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    observations, _ = envs.reset(seed=args.seed)
    q_network = QNetwork(action_dim=envs.single_action_space.n)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(initialization_key, observations),
        target_params=q_network.init(initialization_key, observations),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, 1)
    )

    replay_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        target_values = q_network.apply(q_state.target_params, next_observations)
        target_values = jnp.max(target_values, axis=-1)
        bootstrap_target = rewards + (1 - dones) * args.gamma * target_values

        def loss_function(parameters):
            predictions = q_network.apply(parameters, observations)
            selected_predictions = predictions[jnp.arange(predictions.shape[0]), actions.squeeze()]
            loss = ((selected_predictions - bootstrap_target) ** 2).mean()
            return loss, selected_predictions

        (loss, predictions), gradients = jax.value_and_grad(loss_function, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=gradients)
        return loss, predictions, q_state

    start_time = time.time()
    observations, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.asarray([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            action_values = q_network.apply(q_state.params, observations)
            actions = jax.device_get(action_values.argmax(axis=-1))

        next_observations, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        stored_next_observations = next_observations.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                stored_next_observations[index] = infos["final_observation"][index]

        replay_buffer.add(
            observations,
            stored_next_observations,
            actions,
            rewards,
            terminations,
            infos,
        )
        observations = next_observations

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                batch = replay_buffer.sample(args.batch_size)
                loss, old_values, q_state = update(
                    q_state,
                    batch.observations.numpy(),
                    batch.actions.numpy(),
                    batch.next_observations.numpy(),
                    batch.rewards.flatten().numpy(),
                    batch.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_values).mean(), global_step)
                    steps_per_second = int(global_step / (time.time() - start_time))
                    print("SPS:", steps_per_second)
                    writer.add_scalar("charts/SPS", steps_per_second, global_step)

            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(
                        q_state.params,
                        q_state.target_params,
                        args.tau,
                    )
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as model_file:
            model_file.write(flax.serialization.to_bytes(q_state.params))

        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.dqn_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=run_name,
            Model=QNetwork,
            epsilon=0.05,
        )

        for episode_index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, episode_index)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repository_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(
                args,
                episodic_returns,
                repository_id,
                "DQN",
                f"runs/{run_name}",
                f"videos/{run_name}/eval",
            )

    envs.close()
    writer.close()