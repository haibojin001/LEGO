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
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    env_id: str = "Hopper-v4"
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    policy_noise: float = 0.2
    exploration_noise: float = 0.1
    learning_starts: int = 25e3
    policy_frequency: int = 2
    noise_clip: float = 0.5


def make_env(env_id, seed, idx, capture_video, run_name):
    def create_environment():
        if capture_video and idx == 0:
            environment = gym.make(env_id, render_mode="rgb_array")
            environment = gym.wrappers.RecordVideo(environment, f"videos/{run_name}")
        else:
            environment = gym.make(env_id)
        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment.action_space.seed(seed)
        return environment

    return create_environment


class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        features = jnp.concatenate((x, a), axis=-1)
        features = nn.Dense(256)(features)
        features = nn.relu(features)
        features = nn.Dense(256)(features)
        features = nn.relu(features)
        return nn.Dense(1)(features)


class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        output = nn.Dense(256)(x)
        output = nn.relu(output)
        output = nn.Dense(256)(output)
        output = nn.relu(output)
        output = nn.Dense(self.action_dim)(output)
        output = nn.tanh(output)
        return output * self.action_scale + self.action_bias


class TrainState(FlaxTrainState):
    target_params: flax.core.FrozenDict


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
        % "\n".join(f"|{name}|{value}|" for name, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key, qf2_key = jax.random.split(key, 4)

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    qf = QNetwork()
    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf2_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf2_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf2_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
        key: jnp.ndarray,
    ):
        key, perturbation_key = jax.random.split(key)
        perturbation = jax.random.normal(perturbation_key, actions.shape) * args.policy_noise
        perturbation = jnp.clip(perturbation, -args.noise_clip, args.noise_clip)
        perturbation = perturbation * actor.action_scale

        target_actions = actor.apply(actor_state.target_params, next_observations) + perturbation
        target_actions = jnp.clip(
            target_actions,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )

        q1_target = qf.apply(qf1_state.target_params, next_observations, target_actions).reshape(-1)
        q2_target = qf.apply(qf2_state.target_params, next_observations, target_actions).reshape(-1)
        target_value = rewards + (1 - terminations) * args.gamma * jnp.minimum(q1_target, q2_target)
        target_value = target_value.reshape(-1)

        def critic_loss(parameters):
            estimate = qf.apply(parameters, observations, actions).squeeze()
            loss = jnp.mean((estimate - target_value) ** 2)
            return loss, estimate.mean()

        (loss1, value1), gradient1 = jax.value_and_grad(critic_loss, has_aux=True)(qf1_state.params)
        (loss2, value2), gradient2 = jax.value_and_grad(critic_loss, has_aux=True)(qf2_state.params)

        qf1_state = qf1_state.apply_gradients(grads=gradient1)
        qf2_state = qf2_state.apply_gradients(grads=gradient2)

        return (qf1_state, qf2_state), (loss1, loss2), (value1, value2), key

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        qf2_state: TrainState,
        observations: np.ndarray,
    ):
        def policy_loss(parameters):
            chosen_actions = actor.apply(parameters, observations)
            return -qf.apply(qf1_state.params, observations, chosen_actions).mean()

        loss, gradients = jax.value_and_grad(policy_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=gradients)

        actor_state = actor_state.replace(
            target_params=optax.incremental_update(
                actor_state.params,
                actor_state.target_params,
                args.tau,
            )
        )
        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(
                qf1_state.params,
                qf1_state.target_params,
                args.tau,
            )
        )
        qf2_state = qf2_state.replace(
            target_params=optax.incremental_update(
                qf2_state.params,
                qf2_state.target_params,
                args.tau,
            )
        )

        return (actor_state, qf1_state, qf2_state), loss

    start_time = time.time()

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample()])
        else:
            actions = np.array(actor.apply(actor_state.params, obs))
            actions += np.random.normal(
                0,
                max_action * args.exploration_noise,
                size=envs.action_space.shape,
            )
            actions = np.clip(
                actions,
                envs.single_action_space.low,
                envs.single_action_space.high,
            )

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

        actual_next_obs = next_obs.copy()
        for index, truncated in enumerate(truncations):
            if truncated:
                actual_next_obs[index] = infos["final_observation"][index]

        rb.add(obs, actual_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            (qf1_state, qf2_state), (qf1_loss, qf2_loss), (qf1_values, qf2_values), key = update_critic(
                actor_state,
                qf1_state,
                qf2_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
                key,
            )

            if global_step % args.policy_frequency == 0:
                (actor_state, qf1_state, qf2_state), actor_loss = update_actor(
                    actor_state,
                    qf1_state,
                    qf2_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(actor_state.params))
        print(f"model saved to {model_path}")

        from cleanrl_utils.evals.td3_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Actor,
            exploration_noise=args.exploration_noise,
            seed=args.seed,
            action_dim=np.prod(envs.single_action_space.shape),
            action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
            action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
        )

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_id = f"{args.hf_entity}/{args.env_id}-{args.exp_name}-seed{args.seed}"
            push_to_hub(args, episodic_returns, repo_id, "TD3", run_name)

    envs.close()
    writer.close()