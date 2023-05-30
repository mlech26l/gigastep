from typing import Sequence, Tuple
import numpy as np
import jax.numpy as jnp
import chex
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class ActorCriticMLP(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticLSTM(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

#     @nn.compact
#     def __init__(
#         self,
#         num_hidden_units: int = 32,
#         num_output_units: int = 2,
#     ) -> None:
#         super().__init__(name="LSTM")
#         self.num_hidden_units = num_hidden_units
#         self.num_output_units = num_output_units

#         self._lstm_core = hk.LSTM(self.num_hidden_units)
#         self._policy_head = hk.Linear(self.num_output_units)
#         self._value_head = hk.Linear(1)

#     def initial_state(self, batch_size: int) -> hk.LSTMState:
#         return self._lstm_core.initial_state(batch_size)

#     def __call__(
#         self,
#         x: chex.Array,
#         state: hk.LSTMState,
#     ) -> Tuple[hk.LSTMState, chex.Array, chex.Array]:
#         output, next_state = self._lstm_core(x, state)
#         policy_logits = self._policy_head(output)
#         value = self._value_head(output)

#         return next_state, policy_logits, value
