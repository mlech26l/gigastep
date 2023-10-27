from flax import linen as nn
import jax
import chex
from typing import Tuple, Optional
from evosax.networks.shared import (
    identity_out,
    tanh_out,
    categorical_out,
    gaussian_out,
    default_bias_init,
    kernel_init_fn,
)
from learning.evosax.networks.shared import tanh_gaussian_out


class LSTM(nn.Module):
    """Simple custom LSTM with flexible output head."""

    # LSTM
    num_output_units: int = 10
    num_hidden_units: int = 32
    output_activation: str = "gaussian"
    kernel_init_type: str = "lecun_normal"
    model_name: str = "LSTM"

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        carry: chex.ArrayTree,
        rng: Optional[chex.PRNGKey] = None,
    ) -> Tuple[Tuple[chex.ArrayTree, chex.ArrayTree], chex.Array]:

        info = {}
        
        # Propagate latent state
        lstm_state, x = nn.LSTMCell(
            bias_init=default_bias_init(),
            kernel_init=kernel_init_fn[self.kernel_init_type](),
        )(carry, x)
        if self.output_activation == "identity":
            x = identity_out(x, self.num_output_units, self.kernel_init_type)
        elif self.output_activation == "tanh":
            x = tanh_out(x, self.num_output_units, self.kernel_init_type)
        # Categorical and gaussian output heads require rng for sampling
        elif self.output_activation == "categorical":
            x = categorical_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        elif self.output_activation == "gaussian":
            x = gaussian_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        elif self.output_activation == "tanh_gaussian":
            x, info = tanh_gaussian_out(
                rng, x, self.num_output_units, self.kernel_init_type
            )
        return lstm_state, x, info

    def initialize_carry(self, batch_dims=()) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initialize hidden state of LSTM."""
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, self.num_hidden_units
        )
