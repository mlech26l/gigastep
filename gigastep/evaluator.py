import jax
import jax.numpy as jnp


class EvaluatiorPolicy:
    def apply(self, obs, rng):
        raise NotImplementedError()


class RandomPolicy(EvaluatiorPolicy):
    def __init__(self, env):
        self.env = env
        self.v_apply = jax.vmap(self.apply, in_axes=(0, 0))

    def apply(self, obs, rng):
        if self.env.discrete_actions:
            action = jax.random.randint(
                rng,
                shape=(self.env.n_agents,),
                minval=0,
                maxval=self.env.action_space.n,
            )
        else:
            action = jax.random.uniform(
                rng,
                shape=(self.env.n_agents, self.env.action_space.shape[0]),
                minval=self.env.action_space.low,
                maxval=self.env.action_space.high,
            )
        return action

    def __str__(self):
        return "RandomPolicy"


class CirclePolicy(EvaluatiorPolicy):
    def __init__(self, env, direction=1):
        self.env = env
        action = jnp.array([direction, 0, 0])
        if self.env.discrete_actions:
            action = jnp.argmin(
                jnp.linalg.norm(self.env.action_lut - action[None, :], axis=1)
            )
        self.action = action
        self.direction = direction
        self.v_apply = jax.vmap(self.apply, in_axes=(0, 0))

    def apply(self, obs, rng):
        return self.action

    def __str__(self):
        return f"CirclePolicy{self.direction}"


class CircleRandomPolicy(EvaluatiorPolicy):
    def __init__(self, env):
        self.env = env
        if self.env.discrete_actions:
            raise NotImplementedError()
        self.v_apply = jax.vmap(self.apply, in_axes=(0, 0))

    def apply(self, obs, rng):
        n_agents = obs.shape[0]
        directions = (
            jax.random.randint(
                jax.random.PRNGKey(1), shape=(n_agents,), minval=0, maxval=2
            )
            - 1
        )
        actions = jnp.stack(
            [directions, jnp.zeros(n_agents), jnp.zeros(n_agents)], axis=1
        )
        return actions

    def __str__(self):
        return f"CircleRandomPolicy"


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.policies = [
            RandomPolicy(env),
            CirclePolicy(env),
            CirclePolicy(env, direction=-1),
            CircleRandomPolicy(env),
        ]
        self._ep_rewards = []
        self._ep_alives = []
        self._ep_dones = []
        self.v_merge_actions = jax.vmap(self.merge_actions, in_axes=(0, 0))

        self.team_a_reward = 0
        self.team_b_reward = 0
        self.team_a_wins = 0
        self.team_b_wins = 0
        self.total_games = 0

    @property
    def win_rate_a(self):
        return self.team_a_wins * 100 / self.total_games

    @property
    def win_rate_b(self):
        return self.team_b_wins * 100 / self.total_games

    def __str__(self):
        return (
            f"Team A {self.team_a_wins}/{self.total_games} ({self.team_a_wins*100/self.total_games:0.1f}%) wins [{self.team_a_reward/self.total_games:0.1f} mean return]"
            + f", Team B {self.team_b_wins}/{self.total_games} ({self.team_b_wins*100/self.total_games:0.1f}%) wins [{self.team_b_reward/self.total_games:0.1f} mean return]"
        )

    def merge_actions(self, action1, action2):
        teams = self.env.teams
        if not self.env.discrete_actions:
            teams = teams[:, None]
        return jnp.where(teams == 0, action1, action2)

    def update_episode(self):
        if len(self._ep_rewards) <= 1:
            raise ValueError("No episodes to update. Did you call env.reset()?")
        batch_size = self._ep_rewards[0].shape[0]
        ep_rewards = jnp.stack(self._ep_rewards, axis=1)
        ep_alives = jnp.stack(self._ep_alives, axis=1)
        ep_dones = jnp.stack(self._ep_dones, axis=1)
        # Dimensions: [batch_size, time, n_agents]

        alive_rewards = ep_alives * ep_rewards * (1 - ep_dones[:, :, None])
        alive_rewards = jnp.sum(alive_rewards, axis=1)  # Sum over time

        team_a_rewards = jnp.where(self.env.teams[None, :] == 0, alive_rewards, 0)
        team_b_rewards = jnp.where(self.env.teams[None, :] == 1, alive_rewards, 0)
        self.team_a_reward += team_a_rewards.sum()
        self.team_b_reward += team_b_rewards.sum()

        team_a_alive = ep_alives[:, :, self.env.teams == 0].sum(axis=-1) > 0
        team_b_alive = ep_alives[:, :, self.env.teams == 1].sum(axis=-1) > 0

        team_a_alive = jnp.sum(team_a_alive, axis=1)
        team_b_alive = jnp.sum(team_b_alive, axis=1)

        team_a_wins = team_a_alive > team_b_alive
        team_b_wins = team_b_alive > team_a_alive

        # print("team_a_alive", team_a_alive)
        # print("team_b_alive", team_b_alive)
        # print("team_a_wins", team_a_wins)
        # print("team_b_wins", team_b_wins)
        # print("team_a_wins.sum()", team_a_wins.sum())
        # print("team_b_wins.sum()", team_b_wins.sum())

        self.team_a_wins += team_a_wins.sum()
        self.team_b_wins += team_b_wins.sum()
        self.total_games += batch_size

        self._ep_rewards = []
        self._ep_dones = []
        self._ep_alives = []

    def update_step(self, r, d, ep_done):
        a = jnp.logical_not(d)
        if len(r.shape) == 1:
            # Introduce batch dimension
            r = jnp.expand_dims(r, axis=0)
            a = jnp.expand_dims(a, axis=0)
            ep_done = jnp.expand_dims(ep_done, axis=0)

        # team_a_alive = a[:, self.env.teams == 0].sum(axis=-1)
        # team_b_alive = a[:, self.env.teams == 1].sum(axis=-1)
        # print(f"team_a_alive {team_a_alive}, team_b_alive {team_b_alive}")

        self._ep_rewards.append(r)
        self._ep_alives.append(a)
        self._ep_dones.append(ep_done)