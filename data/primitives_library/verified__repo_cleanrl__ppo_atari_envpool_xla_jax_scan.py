import os
import random
import time
from dataclasses import dataclass
from functools import partial
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
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

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, seed, num_envs):
    def thunk():
        envs = envpool.make(
            env_id,
            env_type="gym",
            num_envs=num_envs,
            episodic_life=True,
            reward_clip=True,
            seed=seed,
        )
        envs.num_envs = num_envs
        envs.single_action_space = envs.action_space
        envs.single_observation_space = envs.observation_space
        envs.is_vector_env = True
        return envs

    return thunk


class Network(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / 255.0
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        return nn.relu(x)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


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
        "|param|value|\n|-|-|\n%s" % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    envs = make_env(args.env_id, args.seed, args.num_envs)()
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    handle, recv, send, step_env = envs.xla()
    handle, (next_obs, _, next_done, _) = recv(handle)

    def step_env_wrappeed(episode_stats, handle, action):
        handle, (next_obs, reward, next_done, info) = step_env(handle, action)
        new_episode_return = episode_stats.episode_returns + info["reward"]
        new_episode_length = episode_stats.episode_lengths + 1
        done = info["terminated"] + info["TimeLimit.truncated"]
        episode_stats = episode_stats.replace(
            episode_returns=new_episode_return * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            episode_lengths=new_episode_length * (1 - info["terminated"]) * (1 - info["TimeLimit.truncated"]),
            returned_episode_returns=jnp.where(done, new_episode_return, episode_stats.returned_episode_returns),
            returned_episode_lengths=jnp.where(done, new_episode_length, episode_stats.returned_episode_lengths),
        )
        return episode_stats, handle, (next_obs, reward, next_done, info)

    network = Network()
    actor = Actor(action_dim=envs.single_action_space.n)
    critic = Critic()

    network_params = network.init(network_key, next_obs)
    hidden = network.apply(network_params, next_obs)
    actor_params = actor.init(actor_key, hidden)
    critic_params = critic.init(critic_key, hidden)
    params = AgentParams(
        network_params=network_params,
        actor_params=actor_params,
        critic_params=critic_params,
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    learning_rate = linear_schedule if args.anneal_lr else args.learning_rate
    train_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        ),
    )

    def get_action_and_value(agent_params, obs, key, action=None):
        hidden = network.apply(agent_params.network_params, obs)
        logits = actor.apply(agent_params.actor_params, hidden)
        value = critic.apply(agent_params.critic_params, hidden).squeeze(-1)
        if action is None:
            action = jax.random.categorical(key, logits)
        log_probs = jax.nn.log_softmax(logits)
        logprob = jnp.take_along_axis(log_probs, action[:, None], axis=1).squeeze(1)
        probs = jax.nn.softmax(logits)
        entropy = -jnp.sum(probs * log_probs, axis=1)
        return action, logprob, entropy, value

    storage = Storage(
        obs=jnp.zeros((args.num_steps,) + next_obs.shape, dtype=next_obs.dtype),
        actions=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.int32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        dones=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        values=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        advantages=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        returns=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
        rewards=jnp.zeros((args.num_steps, args.num_envs), dtype=jnp.float32),
    )

    @jax.jit
    def collect_rollout(train_state, storage, obs, done, episode_stats, handle, key):
        def step(carry, step_index):
            storage, obs, done, episode_stats, handle, key = carry
            key, action_key = jax.random.split(key)
            action, logprob, _, value = get_action_and_value(train_state.params, obs, action_key)
            storage = storage.replace(
                obs=storage.obs.at[step_index].set(obs),
                actions=storage.actions.at[step_index].set(action),
                logprobs=storage.logprobs.at[step_index].set(logprob),
                dones=storage.dones.at[step_index].set(done),
                values=storage.values.at[step_index].set(value),
            )
            episode_stats, handle, (next_obs, reward, next_done, _) = step_env_wrappeed(episode_stats, handle, action)
            storage = storage.replace(rewards=storage.rewards.at[step_index].set(reward))
            return (storage, next_obs, next_done, episode_stats, handle, key), None

        (storage, obs, done, episode_stats, handle, key), _ = jax.lax.scan(
            step,
            (storage, obs, done, episode_stats, handle, key),
            jnp.arange(args.num_steps),
        )
        _, _, _, next_value = get_action_and_value(train_state.params, obs, key)

        def gae_step(carry, transition):
            last_gae, next_value = carry
            done, value, reward = transition
            delta = reward + args.gamma * next_value * (1.0 - done) - value
            last_gae = delta + args.gamma * args.gae_lambda * (1.0 - done) * last_gae
            return (last_gae, value), last_gae

        _, advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(next_value), next_value),
            (storage.dones, storage.values, storage.rewards),
            reverse=True,
        )
        storage = storage.replace(advantages=advantages, returns=advantages + storage.values)
        return storage, obs, done, episode_stats, handle, key

    def loss_fn(params, minibatch):
        _, newlogprob, entropy, newvalue = get_action_and_value(
            params, minibatch.obs, jax.random.PRNGKey(0), minibatch.actions
        )
        logratio = newlogprob - minibatch.logprobs
        ratio = jnp.exp(logratio)

        advantages = minibatch.advantages
        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        newvalue = newvalue.reshape(-1)
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - minibatch.returns) ** 2
            v_clipped = minibatch.values + jnp.clip(newvalue - minibatch.values, -args.clip_coef, args.clip_coef)
            v_loss_clipped = (v_clipped - minibatch.returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - minibatch.returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
        approx_kl = ((ratio - 1.0) - logratio).mean()
        clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > args.clip_coef).astype(jnp.float32))
        return loss, (pg_loss, v_loss, entropy_loss, approx_kl, clipfrac)

    @jax.jit
    def update(train_state, storage, key):
        batch = Storage(
            obs=storage.obs.reshape((args.batch_size,) + storage.obs.shape[2:]),
            actions=storage.actions.reshape(args.batch_size),
            logprobs=storage.logprobs.reshape(args.batch_size),
            dones=storage.dones.reshape(args.batch_size),
            values=storage.values.reshape(args.batch_size),
            advantages=storage.advantages.reshape(args.batch_size),
            returns=storage.returns.reshape(args.batch_size),
            rewards=storage.rewards.reshape(args.batch_size),
        )

        def epoch_step(carry, _):
            train_state, key = carry
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, args.batch_size)
            minibatches = permutation.reshape((args.num_minibatches, args.minibatch_size))

            def minibatch_step(train_state, indices):
                minibatch = jax.tree_util.tree_map(lambda x: x[indices], batch)
                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params, minibatch)
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, metrics

            train_state, metrics = jax.lax.scan(minibatch_step, train_state, minibatches)
            return (train_state, key), metrics

        (train_state, key), metrics = jax.lax.scan(
            epoch_step,
            (train_state, key),
            None,
            length=args.update_epochs,
        )
        return train_state, metrics, key

    global_step = 0
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        storage, next_obs, next_done, episode_stats, handle, key = collect_rollout(
            train_state, storage, next_obs, next_done, episode_stats, handle, key
        )
        global_step += args.batch_size
        train_state, metrics, key = update(train_state, storage, key)

        metrics = jax.device_get(metrics)
        pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = [float(x) for x in metrics.mean(axis=(0, 1))]

        writer.add_scalar("charts/learning_rate", float(linear_schedule(train_state.step)), global_step)
        writer.add_scalar("losses/value_loss", v_loss, global_step)
        writer.add_scalar("losses/policy_loss", pg_loss, global_step)
        writer.add_scalar("losses/entropy", entropy_loss, global_step)
        writer.add_scalar("losses/approx_kl", approx_kl, global_step)
        writer.add_scalar("losses/clipfrac", clipfrac, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        returned_returns = np.asarray(jax.device_get(episode_stats.returned_episode_returns))
        returned_lengths = np.asarray(jax.device_get(episode_stats.returned_episode_lengths))
        for idx, episode_return in enumerate(returned_returns):
            if episode_return != 0:
                writer.add_scalar("charts/episodic_return", episode_return, global_step)
                writer.add_scalar("charts/episodic_length", returned_lengths[idx], global_step)

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    if args.save_model:
        from cleanrl_utils.evals.ppo_eval import evaluate

        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(train_state.params))

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Network, Actor, Critic),
            device=None,
            capture_video=args.capture_video,
            seed=args.seed,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", video_folder_path=f"videos/{run_name}-eval")

    envs.close()
    writer.close()