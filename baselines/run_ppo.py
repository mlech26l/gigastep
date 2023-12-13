import distrax
from gigastep import make_scenario
import jax
import jax.numpy as jnp
import optax
from baselines.bline_ppo import make_random_adversary, RolloutManager, Runner

from flax.training.train_state import TrainState
import flax.linen as nn
from typing import Sequence


class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        pi = distrax.Categorical(logits=x)
        return pi


class ValueMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x


def test_rollout():
    env = make_scenario("identical_5_vs_5_fobs_vec_void_disc")
    rng = jax.random.PRNGKey(3)

    model = MLP([128, env.n_actions])
    policy_state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, jnp.zeros((1,) + env.observation_space.shape)),
        tx=optax.adam(1e-3),
    )

    value_model = ValueMLP([128, 1])
    value_state = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(rng, jnp.zeros((1,) + env.observation_space.shape)),
        tx=optax.adam(1e-3),
    )

    config = {
        "value_clip_eps": 0.2,
        "entropy_coef": 0.01,
        "clip_eps": 0.2,
        "epochs_per_iter": 4,
        "ppo_iters": 200,
        "train_batch_size": 128,
        "rollout_batch_size": 1024,
        "gae_lambda": 1.0,
        "gamma": 0.99,
    }
    runner = Runner(env, policy_state, value_state, config)
    runner.run()


if __name__ == "__main__":
    test_rollout()
