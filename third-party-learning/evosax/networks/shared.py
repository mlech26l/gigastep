import jax
import jax.numpy as jnp
from flax import linen as nn
import chex
from jax.nn.initializers import (
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
    xavier_normal,
    xavier_uniform,
)


kernel_init_fn = {
    "glorot_normal": glorot_normal,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "he_uniform": he_uniform,
    "kaiming_normal": kaiming_normal,
    "kaiming_uniform": kaiming_uniform,
    "lecun_normal": lecun_normal,
    "lecun_uniform": lecun_uniform,
    "xavier_normal": xavier_normal,
    "xavier_uniform": xavier_uniform,
}


def default_bias_init(scale=0.05):
    return nn.initializers.uniform(scale)


def tanh_gaussian_out(
    rng: chex.PRNGKey,
    x: chex.Array,
    num_output_units: int,
    init_type: str = "lecun_normal",
) -> chex.Array:
    """Simple affine layers for mean and log var + gaussian sample."""
    x_mean = nn.Dense(
        features=num_output_units,
        kernel_init=kernel_init_fn[init_type](),
        bias_init=default_bias_init(),
    )(x)
    x_log_var = nn.Dense(
        features=num_output_units,  # 1,
        kernel_init=kernel_init_fn[init_type](),
        bias_init=default_bias_init(scale=1.0),
    )(x)
    x_std = jnp.exp(0.5 * x_log_var)
    noise = x_std * jax.random.normal(rng, (num_output_units,))
    info = {}
    info['act_mean'] = x_mean
    info['act_stdv'] = noise
    # return 1.1*nn.tanh(x_mean) + noise, info
    return x_mean + noise, info
