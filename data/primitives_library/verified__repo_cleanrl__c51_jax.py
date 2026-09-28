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
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
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

    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_atoms: int = 101
    """the number of atoms"""
    v_min: float = -100
    """the return lower bound"""
    v_max: float = 100
    """the return upper bound"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""


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
    n_atoms: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.n_atoms)(x)
        x = x.reshape((x.shape[0], self.action_dim, self.n_atoms))
        return nn.softmax(x, axis=-1)


class TrainState(FlaxTrainState):
    target_params: flax.core.FrozenDict
    atoms: jnp.ndarray


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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
        % ("\n".join(f"|{key}|{value}|" for key, value in vars(args).items())),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
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
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        target_distributions = q_network.apply(q_state.target_params, next_observations)
        target_values = (target_distributions * q_state.atoms).sum(axis=-1)
        greedy_actions = jnp.argmax(target_values, axis=-1)
        target_distributions = target_distributions[np.arange(target_distributions.shape[0]), greedy_actions]

        shifted_atoms = rewards + args.gamma * q_state.atoms * (1 - dones)
        spacing = q_state.atoms[1] - q_state.atoms[0]
        shifted_atoms = jnp.clip(shifted_atoms, a_min=args.v_min, a_max=args.v_max)

        bin_locations = (shifted_atoms - args.v_min) / spacing
        lower_bins = jnp.clip(jnp.floor(bin_locations), a_min=0, a_max=args.n_atoms - 1)
        upper_bins = jnp.clip(jnp.ceil(bin_locations), a_min=0, a_max=args.n_atoms - 1)

        lower_mass = (upper_bins + (lower_bins == upper_bins).astype(jnp.float32) - bin_locations) * target_distributions
        upper_mass = (bin_locations - lower_bins) * target_distributions
        projected_distribution = jnp.zeros_like(target_distributions)

        def add_projection(index, distribution):
            distribution = distribution.at[index, lower_bins[index].astype(jnp.int32)].add(lower_mass[index])
            distribution = distribution.at[index, upper_bins[index].astype(jnp.int32)].add(upper_mass[index])
            return distribution

        projected_distribution = jax.lax.fori_loop(
            0,
            projected_distribution.shape[0],
            add_projection,
            projected_distribution,
        )

        def loss_fn(params, observations, actions, targets):
            distributions = q_network.apply(params, observations)
            selected_distributions = distributions[np.arange(distributions.shape[0]), actions.squeeze()]
            clipped_distributions = jnp.clip(selected_distributions, a_min=1e-5, a_max=1 - 1e-5)
            cross_entropy = (-(targets * jnp.log(clipped_distributions)).sum(-1)).mean()
            values = (selected_distributions * q_state.atoms).sum(-1)
            return cross_entropy, values

        (loss_value, old_values), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            q_state.params,
            observations,
            actions,
            projected_distribution,
        )
        q_state = q_state.apply_gradients(grads=gradients)
        return loss_value, old_values, q_state

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            pmfs = q_network.apply(q_state.params, obs)
            values = (pmfs * q_state.atoms).sum(axis=-1)
            actions = jax.device_get(values.argmax(axis=-1))

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[index] = infos["final_observation"][index]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                loss, old_values, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.numpy(),
                    data.dones.numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_values.mean(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, 1)
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.c51_jax_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(QNetwork, {"action_dim": envs.single_action_space.n, "n_atoms": args.n_atoms}),
            epsilon=0.05,
            atoms=q_state.atoms,
        )

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(args, episodic_returns, repo_id, "C51", f"runs/{run_name}", video_fps=30)

    envs.close()
    writer.close()