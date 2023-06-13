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
from evosax.networks.cnn import CNN


class LSTM_CNN(nn.Module):
    """Simple LSTM Wrapper with CNN encoder and flexible output head."""

    # CNN
    num_output_units_cnn: int = 10
    depth_1: int = 1
    depth_2: int = 1
    features_1: int = 8
    features_2: int = 16
    kernel_1: int = 5
    kernel_2: int = 5
    strides_1: int = 1
    strides_2: int = 1
    num_linear_layers_cnn: int = 1
    num_hidden_units_cnn: int = 16
    hidden_activation_cnn: str = "relu"
    output_activation_cnn: str = "identity"
    kernel_init_type_cnn: str = "lecun_normal"

    encoder: CNN = CNN(
        num_output_units=num_hidden_units_cnn,
        depth_1=depth_1,
        depth_2=depth_2,
        features_1=features_1,
        features_2=features_2,
        kernel_1=kernel_1,
        kernel_2=kernel_2,
        strides_1=strides_1,
        strides_2=strides_2,
        num_linear_layers=num_linear_layers_cnn,
        num_hidden_units=num_hidden_units_cnn,
        hidden_activation=hidden_activation_cnn,
        output_activation=output_activation_cnn,
        kernel_init_type=kernel_init_type_cnn,
    )

    # LSTM
    num_output_units_lstm: int = 10
    num_hidden_units_lstm: int = 32
    output_activation_lstm: str = "gaussian"
    kernel_init_type_lstm: str = "lecun_normal"
    model_name: str = "LSTM_CNN"

    @nn.compact
    def __call__(
        self,
        x: chex.Array,
        carry: chex.ArrayTree,
        rng: Optional[chex.PRNGKey] = None,
    ) -> Tuple[Tuple[chex.ArrayTree, chex.ArrayTree], chex.Array]:

        info = {}

        rng_cnn, rng_lstm = jax.random.split(rng, 2)

        # Encode observation
        x = self.encoder(x, rng_cnn)

        # Propagate latent state
        lstm_state, x = nn.LSTMCell(
            bias_init=default_bias_init(),
            kernel_init=kernel_init_fn[self.kernel_init_type_lstm](),
        )(carry, x)
        if self.output_activation_lstm == "identity":
            x = identity_out(x, self.num_output_units_lstm, self.kernel_init_type_lstm)
        elif self.output_activation_lstm == "tanh":
            x = tanh_out(x, self.num_output_units_lstm, self.kernel_init_type_lstm)
        # Categorical and gaussian output heads require rng for sampling
        elif self.output_activation_lstm == "categorical":
            x = categorical_out(
                rng_lstm, x, self.num_output_units_lstm, self.kernel_init_type_lstm
            )
        elif self.output_activation_lstm == "gaussian":
            x = gaussian_out(
                rng_lstm, x, self.num_output_units_lstm, self.kernel_init_type_lstm
            )
        elif self.output_activation_lstm == "tanh_gaussian":
            x, info = tanh_gaussian_out(
                rng_lstm, x, self.num_output_units_lstm, self.kernel_init_type_lstm
            )
        return lstm_state, x, info

    def initialize_carry(self, batch_dims=()) -> Tuple[chex.ArrayTree, chex.ArrayTree]:
        """Initialize hidden state of LSTM."""
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, self.num_hidden_units_lstm
        )
