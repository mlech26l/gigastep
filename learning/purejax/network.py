from typing import Sequence, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import chex
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class ActorCriticMLP(nn.Module):
    action_dim: Sequence[int]
    teams: jnp.ndarray
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if len(x.shape) == 2:
            team1 = (1 - self.teams)[:, None]
            team2 = self.teams[:, None]
        else: # with num_env dim
            team1 = (1 - self.teams)[None, :, None]
            team2 = self.teams[None, :, None]
        
        actor_mean_team1 = self._infer_actor(x, activation, scope="actor/team1/")
        actor_mean_team2 = self._infer_actor(x, activation, scope="actor/team2/")
        # actor_mean_team2 = jnp.zeros_like(actor_mean_team1) # HACK: fixed behavior
        actor_mean = actor_mean_team1 * team1 + actor_mean_team2 * team2
        pi = distrax.Categorical(logits=actor_mean)

        critic_team1 = self._infer_critic(x, activation, team1, scope="critic/team1/")
        critic_team2 = self._infer_critic(x, activation, team2, scope="critic/team2/")
        critic = critic_team1 * team1 + critic_team2 * team2

        return pi, jnp.squeeze(critic, axis=-1)
    
    def _infer_actor(self, x, activation, scope=""):
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_0",
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_1",
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name=scope+"dense_2",
        )(actor_mean)

        return actor_mean

    def _infer_critic(self, x, activation, team, scope=""):
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_0",
        )(x)
        critic = activation(critic)
        critic = self._mix_team_critic(critic, team)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_1",
        )(critic)
        critic = activation(critic)
        critic = self._mix_team_critic(critic, team)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name=scope+"dense_2",
        )(critic)

        return critic

    def _mix_team_critic(self, critic, team):
        team_critic = (critic * team).sum(-2, keepdims=True) / team.sum(-2, keepdims=True)
        team_critic = jnp.repeat(team_critic, critic.shape[-2], axis=-2)
        critic = jnp.concatenate([critic, team_critic], axis=-1)

        return critic


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
