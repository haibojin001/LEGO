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
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """the seed of the experiment"""
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

    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25e3
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""


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
        features = nn.Dense(256)(x)
        features = nn.relu(features)
        features = nn.Dense(256)(features)
        features = nn.relu(features)
        features = nn.Dense(self.action_dim)(features)
        features = nn.tanh(features)
        return features * self.action_scale + self.action_bias


class TrainState(TrainState):
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
        % "\n".join(f"|{key}|{value}|" for key, value in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

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

    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        next_actions = actor.apply(actor_state.target_params, next_observations).clip(-1, 1)
        next_values = qf.apply(qf1_state.target_params, next_observations, next_actions).reshape(-1)
        target_values = (rewards + (1 - terminations) * args.gamma * next_values).reshape(-1)

        def critic_loss(params):
            predicted_values = qf.apply(params, observations, actions).squeeze()
            loss = ((predicted_values - target_values) ** 2).mean()
            return loss, predicted_values.mean()

        (loss_value, average_value), gradients = jax.value_and_grad(
            critic_loss, has_aux=True
        )(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=gradients)
        return qf1_state, loss_value, average_value

    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def policy_loss(params):
            selected_actions = actor.apply(params, observations)
            return -qf.apply(qf1_state.params, observations, selected_actions).mean()

        loss_value, gradients = jax.value_and_grad(policy_loss)(actor_state.params)
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
        return actor_state, qf1_state, loss_value

    start_time = time.time()
    actor_loss_value = jnp.array(0.0)
    qf1_loss_value = jnp.array(0.0)
    qf1_a_values = jnp.array(0.0)

    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            policy_actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (
                        jax.device_get(policy_actions)[0]
                        + np.random.normal(
                            0,
                            actor.action_scale * args.exploration_noise,
                        )[0]
                    ).clip(
                        envs.single_action_space.low,
                        envs.single_action_space.high,
                    )
                ]
            )

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        real_next_obs = next_obs.copy()
        for idx, truncated in enumerate(truncations):
            if truncated:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.numpy().flatten(),
                data.dones.numpy().flatten(),
            )

            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                sps = int(global_step / (time.time() - start_time))
                writer.add_scalar("charts/SPS", sps, global_step)
                print("SPS:", sps)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as file:
            file.write(flax.serialization.to_bytes(actor_state.params))
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()