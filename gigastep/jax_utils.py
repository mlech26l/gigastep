import jax
import optax
import jax.numpy as jnp
from flax.training import train_state


class Box:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    @property
    def shape(self):
        return self.low.shape

    @property
    def dtype(self):
        return self.low.dtype

    # Jax function that samples uniformly from the box
    def sample(self, rng):
        return (
            jax.random.uniform(rng, self.low.shape, self.low.dtype)
            * (self.high - self.low)
            + self.low
        )


class Discrete:
    def __init__(self, n):
        self.n = n

    @property
    def shape(self):
        return ()

    @property
    def dtype(self):
        return jnp.int32

    # Jax function that samples uniformly from the multinoulli
    def sample(self, rng):
        return jax.random.randint(rng, (), 0, self.n)


def create_train_state(model, rng, in_dim, learning_rate, ema=0, clip_norm=None):
    """Creates initial `TrainState`."""
    params = model.init(rng, jnp.ones([1, in_dim]))
    tx = optax.adam(learning_rate)
    if clip_norm is not None:
        tx = optax.chain(tx, optax.clip_by_global_norm(clip_norm))
    if ema > 0:
        tx = optax.chain(tx, optax.ema(ema))
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)