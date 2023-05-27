import jax
import jax.numpy as jnp


class EvaluatiorPolicy:
    def apply(self, obs, rng):
        raise NotImplementedError()


class RandomPolicy(EvaluatiorPolicy):
    def __init__(self, env):
        self.env = env

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
                shape=(self.env.n_agents, self.env.action_space.shape),
                minval=self.env.action_space.low,
                maxval=self.env.action_space.high,
            )
        return action


class CirclePolicy(EvaluatiorPolicy):
    def __init__(self, env):
        self.env = env

    def apply(self, obs, rng):
        action = jnp.array([1, 0, 0])
        if self.env.discrete_actions:
            action = jnp.argmin(
                jnp.linalg.norm(self.env.action_lut - action[None, :], axis=1)
            )
        return action

def get_empty_reward_counter(batch_size):


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.policies = [RandomPolicy(env), CirclePolicy(env)]
        self._ep_rewards = []
        self._ep_alives = []
        self._ep_dones = []

        self.team_a_reward = 0
        self.team_b_reward = 0
        self.team_a_wins = 0
        self.team_b_wins = 0
        self.total_games = 0

    def merge_actions(self, action1, action2):
        return jnp.where(self.env.per_agent_team == 0, action1, action2)

    # TODO: Make this vectorized
    # TODO: Support batched and unbatched
    # TODO: open + close episodes -> defer reward computation to the end to not double count

    def end_of_episode(self):


    def update_reward(self, r, a, d):
        batch_mode = len(r.shape) > 1
        self.team_a_reward += jnp.sum(r * a * (1 - self.env.per_agent_team))
        self.team_b_reward += jnp.sum(r * a * self.env.per_agent_team)

        team_a_wins = jnp.sum(self.env.per_agent_team * a, axis=-1)
        self.team_a_wins += jnp.sum(r * (1 - self.env.per_agent_team) > 0)
        self.team_b_wins += jnp.sum(r * self.env.per_agent_team > 0)
        self.total_games += self.env.n_agents

    def __iter__(self):
        for policy in self.policies:
            yield policy