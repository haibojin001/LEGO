import os
import random
import time
from dataclasses import dataclass

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"

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
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "BreakoutNoFrameskip-v4"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    n_atoms: int = 51
    v_min: float = -10
    v_max: float = 10
    buffer_size: int = 1000000
    gamma: float = 0.99
    target_network_frequency: int = 10000
    batch_size: int = 32
    start_e: float = 1
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    learning_starts: int = 80000
    train_frequency: int = 4


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


class QNetwork(nn.Module):
    action_dim: int
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x.astype(jnp.float32) / 255.0
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return nn.softmax(x, axis=-1)


class TrainState(FlaxTrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    rate = (end_e - start_e) / duration
    return max(start_e + rate * t, end_e)


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
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs, _ = envs.reset(seed=args.seed)

    q_network = QNetwork(action_dim=envs.single_action_space.n, n_atoms=args.n_atoms)
    initial_params = q_network.init(q_key, obs)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=initial_params,
        target_params=initial_params,
        atoms=jnp.asarray(np.linspace(args.v_min, args.v_max, num=args.n_atoms)),
        tx=optax.adam(learning_rate=args.learning_rate, eps=0.01 / args.batch_size),
    )

    q_network.apply = jax.jit(q_network.apply)
    q_state = q_state.replace(
        target_params=optax.incremental_update(q_state.params, q_state.target_params, 1)
    )

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        next_pmfs = q_network.apply(q_state.target_params, next_observations)
        next_q_values = (next_pmfs * q_state.atoms).sum(axis=-1)
        selected_next_actions = jnp.argmax(next_q_values, axis=-1)
        next_pmfs = next_pmfs[jnp.arange(next_pmfs.shape[0]), selected_next_actions]

        shifted_atoms = rewards + args.gamma * q_state.atoms * (1.0 - dones)
        atom_gap = q_state.atoms[1] - q_state.atoms[0]
        clipped_atoms = jnp.clip(shifted_atoms, args.v_min, args.v_max)
        positions = (clipped_atoms - args.v_min) / atom_gap
        lower = jnp.clip(jnp.floor(positions), 0, args.n_atoms - 1)
        upper = jnp.clip(jnp.ceil(positions), 0, args.n_atoms - 1)

        lower_mass = (upper + (lower == upper).astype(jnp.float32) - positions) * next_pmfs
        upper_mass = (positions - lower) * next_pmfs
        projected = jnp.zeros_like(next_pmfs)

        def scatter_row(i, result):
            result = result.at[i, lower[i].astype(jnp.int32)].add(lower_mass[i])
            result = result.at[i, upper[i].astype(jnp.int32)].add(upper_mass[i])
            return result

        projected = jax.lax.fori_loop(0, projected.shape[0], scatter_row, projected)

        def objective(parameters, batch_observations, batch_actions, target_distribution):
            pmfs = q_network.apply(parameters, batch_observations)
            chosen_pmfs = pmfs[jnp.arange(pmfs.shape[0]), batch_actions.squeeze()]
            safe_pmfs = jnp.clip(chosen_pmfs, 1e-5, 1.0 - 1e-5)
            cross_entropy = -(target_distribution * jnp.log(safe_pmfs)).sum(axis=-1)
            return cross_entropy.mean(), (chosen_pmfs * q_state.atoms).sum(axis=-1)

        (loss_value, old_values), gradients = jax.value_and_grad(objective, has_aux=True)(
            q_state.params,
            observations,
            actions,
            projected,
        )
        q_state = q_state.apply_gradients(grads=gradients)
        return loss_value, old_values, q_state

    start_time = time.time()

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
            pmfs = q_network.apply(q_state.params, jnp.asarray(obs))
            q_values = (pmfs * q_state.atoms).sum(axis=-1)
            actions = np.asarray(jnp.argmax(q_values, axis=-1))

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
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
        for i, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[i] = infos["final_observation"][i]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                loss, old_values, q_state = update(
                    q_state,
                    jnp.asarray(data.observations),
                    jnp.asarray(data.actions),
                    jnp.asarray(data.next_observations),
                    jnp.asarray(data.rewards),
                    jnp.asarray(data.dones),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", float(loss), global_step)
                    writer.add_scalar("losses/q_values", float(old_values.mean()), global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    print(
                        "SPS:",
                        int(global_step / (time.time() - start_time)),
                    )

            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(target_params=q_state.params)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.c51_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            epsilon=0.05,
            capture_video=args.capture_video,
            seed=args.seed,
            n_atoms=args.n_atoms,
            v_min=args.v_min,
            v_max=args.v_max,
        )

        for index, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, index)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            if args.hf_entity:
                repo_id = f"{args.hf_entity}/{repo_id}"
            push_to_hub(args, episodic_returns, repo_id, "C51", model_path)

    envs.close()
    writer.close()