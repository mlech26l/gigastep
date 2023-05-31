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
    has_cnn: bool = False
    obs_shape: Sequence[int] = None

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        if self.has_cnn:
            if len(x.shape) < 4: # convert flattened image obs to image
                x = x.reshape(x.shape[:-1] + self.obs_shape[:-1] + (-1,))

            if len(x.shape) == 4:
                team1 = (1 - self.teams)[:, None]
                team2 = self.teams[:, None]
                team1_for_cnn = (1 - self.teams)[:, None, None, None]
                team2_for_cnn = self.teams[:, None, None, None]
            else: # with num_env dim
                team1 = (1 - self.teams)[None, :, None]
                team2 = self.teams[None, :, None]
                team1_for_cnn = (1 - self.teams)[None, :, None, None, None]
                team2_for_cnn = self.teams[None, :, None, None, None]
        else:
            if len(x.shape) == 2:
                team1 = (1 - self.teams)[:, None]
                team2 = self.teams[:, None]
            else: # with num_env dim
                team1 = (1 - self.teams)[None, :, None]
                team2 = self.teams[None, :, None]
            team1_for_cnn = team1
            team2_for_cnn = team2
        
        actor_mean_team1 = self._infer_actor(x, activation, scope="actor/team1/")
        actor_mean_team2 = self._infer_actor(x, activation, scope="actor/team2/")
        # actor_mean_team2 = jnp.zeros_like(actor_mean_team1) # HACK: fixed behavior
        actor_mean = actor_mean_team1 * team1 + actor_mean_team2 * team2
        pi = distrax.Categorical(logits=actor_mean)

        critic_team1 = self._infer_critic(x, activation, team1, team1_for_cnn, scope="critic/team1/")
        critic_team2 = self._infer_critic(x, activation, team1, team2_for_cnn, scope="critic/team2/")
        critic = critic_team1 * team1 + critic_team2 * team2

        return pi, jnp.squeeze(critic, axis=-1)
    
    def _infer_actor(self, x, activation, scope=""):
        if self.has_cnn:
            actor_mean = nn.Conv(features=64, kernel_size=(3, 3))(x)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.max_pool(actor_mean, window_shape=(2, 2), strides=(2, 2))
            actor_mean = nn.Conv(features=64, kernel_size=(3, 3))(actor_mean)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.max_pool(actor_mean, window_shape=(2, 2), strides=(2, 2))
            actor_mean = nn.Conv(features=64, kernel_size=(3, 3))(actor_mean)
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.max_pool(actor_mean, window_shape=(2, 2), strides=(2, 2))
            if True:
                actor_mean = nn.Conv(features=256, kernel_size=(1, 1))(actor_mean)
                actor_mean = actor_mean.mean(-2).mean(-2)
            else:
                actor_mean = actor_mean.reshape(actor_mean.shape[:-3] + (np.prod(actor_mean.shape[-3:]),))
                actor_mean = nn.Dense(features=1024)(actor_mean)
            actor_mean = activation(actor_mean)
        else:
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_0",
            )(x)
            actor_mean = activation(actor_mean)
            actor_mean = nn.Dense(
                64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name=scope+"dense_1",
            )(actor_mean)
            actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name=scope+"last",
        )(actor_mean)

        return actor_mean

    def _infer_critic(self, x, activation, team, team_for_cnn, scope=""):
        if self.has_cnn:
            critic = nn.Conv(features=64, kernel_size=(3, 3))(x)
            critic = nn.relu(critic)
            critic = nn.max_pool(critic, window_shape=(2, 2), strides=(2, 2))
            critic = self._mix_team_critic(critic, team_for_cnn, for_cnn=True)
            critic = nn.Conv(features=64, kernel_size=(3, 3))(critic)
            critic = nn.relu(critic)
            critic = nn.max_pool(critic, window_shape=(2, 2), strides=(2, 2))
            critic = self._mix_team_critic(critic, team_for_cnn, for_cnn=True)
            critic = nn.Conv(features=64, kernel_size=(3, 3))(critic)
            critic = nn.relu(critic)
            critic = nn.max_pool(critic, window_shape=(2, 2), strides=(2, 2))
            critic = self._mix_team_critic(critic, team_for_cnn, for_cnn=True)
            if True:
                critic = nn.Conv(features=256, kernel_size=(1, 1))(critic)
                critic = critic.mean(-2).mean(-2)
            else:
                critic = critic.reshape(critic.shape[:-3] + (np.prod(critic.shape[-3:]),))
                critic = nn.Dense(features=1024)(critic)
            critic = activation(critic)
            critic = self._mix_team_critic(critic, team)
        else:
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
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name=scope+"last",
        )(critic)

        return critic

    def _mix_team_critic(self, critic, team, for_cnn=False):
        if for_cnn:
            team_critic = (critic * team).sum(-4, keepdims=True) / team.sum(-4, keepdims=True)
            team_critic = jnp.repeat(team_critic, critic.shape[-4], axis=-4)
        else:
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
