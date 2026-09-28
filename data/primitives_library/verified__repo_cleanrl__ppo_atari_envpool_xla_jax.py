import os
import random
import time
from dataclasses import dataclass
from typing import Sequence

import envpool
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False

    env_id: str = "Breakout-v5"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=512,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        return nn.relu(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)


@flax.struct.dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


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
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    handle, recv, send, step_env = envs.xla()

    def step_env_wrappeed(episode_statistics, current_handle, action):
        current_handle, (next_observation, reward, next_done, info) = step_env(current_handle, action)

        running_return = episode_statistics.episode_returns + info["reward"]
        running_length = episode_statistics.episode_lengths + 1
        finished = info["terminated"] + info["TimeLimit.truncated"]

        episode_statistics = episode_statistics.replace(
            episode_returns=running_return * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=running_length * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            returned_episode_returns=jnp.where(
                finished,
                running_return,
                episode_statistics.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                finished,
                running_length,
                episode_statistics.returned_episode_lengths,
            ),
        )
        return episode_statistics, current_handle, (next_observation, reward, next_done, info)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    def linear_schedule(count):
        fraction = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * fraction

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()

    observation_example = np.array([envs.single_observation_space.sample()])
    network_params = network.init(network_key, observation_example)
    hidden_example = network.apply(network_params, observation_example)
    actor_params = actor.init(actor_key, hidden_example)
    critic_params = critic.init(critic_key, hidden_example)

    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(
            learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
            eps=1e-5,
        ),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=AgentParams(
            network_params=network_params,
            actor_params=actor_params,
            critic_params=critic_params,
        ),
        tx=optimizer,
    )

    def get_action_and_value(state, observation, rng):
        hidden = network.apply(state.params.network_params, observation)
        logits = actor.apply(state.params.actor_params, hidden)
        values = critic.apply(state.params.critic_params, hidden).squeeze(-1)
        action = jax.random.categorical(rng, logits)
        logprobabilities = jax.nn.log_softmax(logits)
        logprobability = jnp.take_along_axis(logprobabilities, action[:, None], axis=1).squeeze(1)
        entropy = -(jax.nn.softmax(logits) * logprobabilities).sum(axis=1)
        return action, logprobability, entropy, values

    def get_value(state, observation):
        hidden = network.apply(state.params.network_params, observation)
        return critic.apply(state.params.critic_params, hidden).squeeze(-1)

    @jax.jit
    def collect_rollout(state, initial_observation, initial_done, initial_handle, statistics, rng):
        def one_step(carry, _):
            state_, observation, done, handle_, statistics_, rng_ = carry
            rng_, action_rng = jax.random.split(rng_)
            action, logprob, _, value = get_action_and_value(state_, observation, action_rng)
            statistics_, handle_, (next_observation, reward, next_done, _) = step_env_wrappeed(
                statistics_, handle_, action
            )

            item = Storage(
                obs=observation,
                actions=action,
                logprobs=logprob,
                dones=done,
                values=value,
                advantages=jnp.zeros_like(reward),
                returns=jnp.zeros_like(reward),
                rewards=reward,
            )
            return (state_, next_observation, next_done, handle_, statistics_, rng_), item

        carry, storage = jax.lax.scan(
            one_step,
            (state, initial_observation, initial_done, initial_handle, statistics, rng),
            None,
            length=args.num_steps,
        )
        state, next_observation, next_done, handle, statistics, rng = carry
        next_value = get_value(state, next_observation)

        def gae_step(carry, transition):
            gae, subsequent_value, subsequent_done = carry
            done, value, reward = transition
            td_error = reward + args.gamma * subsequent_value * (1.0 - subsequent_done) - value
            gae = td_error + args.gamma * args.gae_lambda * (1.0 - subsequent_done) * gae
            return (gae, value, done), gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(next_value), next_value, next_done),
            (storage.dones, storage.values, storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return state, next_observation, next_done, handle, statistics, rng, storage

    @jax.jit
    def update_agent(state, storage, rng):
        flattened_storage = jax.tree_util.tree_map(
            lambda value: value.reshape((args.batch_size,) + value.shape[2:]),
            storage,
        )

        def minibatch_update(carry, indices):
            state_, rng_ = carry
            minibatch = jax.tree_util.tree_map(lambda value: value[indices], flattened_storage)

            def loss_function(parameters):
                hidden = network.apply(parameters.network_params, minibatch.obs)
                logits = actor.apply(parameters.actor_params, hidden)
                values = critic.apply(parameters.critic_params, hidden).squeeze(-1)

                log_probabilities = jax.nn.log_softmax(logits)
                new_logprobs = jnp.take_along_axis(
                    log_probabilities,
                    minibatch.actions[:, None],
                    axis=1,
                ).squeeze(1)
                entropy = -(jax.nn.softmax(logits) * log_probabilities).sum(axis=1)

                logratio = new_logprobs - minibatch.logprobs
                ratio = jnp.exp(logratio)

                if args.norm_adv:
                    advantages = (minibatch.advantages - minibatch.advantages.mean()) / (
                        minibatch.advantages.std() + 1e-8
                    )
                else:
                    advantages = minibatch.advantages

                unclipped_policy_loss = -advantages * ratio
                clipped_policy_loss = -advantages * jnp.clip(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                policy_loss = jnp.maximum(unclipped_policy_loss, clipped_policy_loss).mean()

                if args.clip_vloss:
                    clipped_values = minibatch.values + jnp.clip(
                        values - minibatch.values,
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    value_loss = 0.5 * jnp.maximum(
                        (values - minibatch.returns) ** 2,
                        (clipped_values - minibatch.returns) ** 2,
                    ).mean()
                else:
                    value_loss = 0.5 * ((values - minibatch.returns) ** 2).mean()

                entropy_loss = entropy.mean()
                total_loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss
                approximate_kl = ((ratio - 1.0) - logratio).mean()
                clip_fraction = (jnp.abs(ratio - 1.0) > args.clip_coef).mean()

                return total_loss, (
                    policy_loss,
                    value_loss,
                    entropy_loss,
                    approximate_kl,
                    clip_fraction,
                )

            (_, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(state_.params)
            state_ = state_.apply_gradients(grads=gradients)
            return (state_, rng_), metrics

        def epoch_update(carry, _):
            state_, rng_ = carry
            rng_, permutation_rng = jax.random.split(rng_)
            permutation = jax.random.permutation(permutation_rng, args.batch_size)
            minibatches = permutation.reshape((args.num_minibatches, args.minibatch_size))
            (state_, rng_), metrics = jax.lax.scan(minibatch_update, (state_, rng_), minibatches)
            return (state_, rng_), metrics

        (state, rng), metrics = jax.lax.scan(
            epoch_update,
            (state, rng),
            None,
            length=args.update_epochs,
        )
        return state, rng, metrics

    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs, dtype=jnp.float32)
    next_obs = jnp.asarray(next_obs)
    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        (
            agent_state,
            next_obs,
            next_done,
            handle,
            episode_stats,
            key,
            storage,
        ) = collect_rollout(
            agent_state,
            next_obs,
            next_done,
            handle,
            episode_stats,
            key,
        )

        global_step += args.batch_size
        agent_state, key, loss_metrics = update_agent(agent_state, storage, key)

        policy_loss = float(loss_metrics[..., 0].mean())
        value_loss = float(loss_metrics[..., 1].mean())
        entropy_loss = float(loss_metrics[..., 2].mean())
        approximate_kl = float(loss_metrics[..., 3].mean())
        clip_fraction = float(loss_metrics[..., 4].mean())

        writer.add_scalar(
            "charts/learning_rate",
            linear_schedule(agent_state.step) if args.anneal_lr else args.learning_rate,
            global_step,
        )
        writer.add_scalar(
            "charts/episodic_return",
            float(episode_stats.returned_episode_returns.mean()),
            global_step,
        )
        writer.add_scalar(
            "charts/episodic_length",
            float(episode_stats.returned_episode_lengths.mean()),
            global_step,
        )
        writer.add_scalar("losses/value_loss", value_loss, global_step)
        writer.add_scalar("losses/policy_loss", policy_loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss, global_step)
        writer.add_scalar("losses/approx_kl", approximate_kl, global_step)
        writer.add_scalar("losses/clipfrac", clip_fraction, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()