import jax
import jax.numpy as jnp
import torch
import sys
import time

import numpy as np

from gigastep import GigastepViewer 

SLEEP_TIME = 0.01


class EvaluatiorPolicy:
    def apply(self, obs, rng):
        raise NotImplementedError()

class Torch_Policy(EvaluatiorPolicy):
    def __init__(self, env, policy, device = "cpu", vectorize = False, switch_side = False):
        self.actor_critic = policy
        if vectorize:
            self.recurrent_hidden_states = torch.zeros(env.n_agents,
                          policy.recurrent_hidden_state_size,
                          device=device
            ) if policy is not None else None
        else:
            self.recurrent_hidden_states = torch.zeros(env.n_agents,
                              policy.recurrent_hidden_state_size,
                              device=device
                ) if policy is not None else None
        self.n_ego_agents = env.n_teams[0]
        self.n_opp_agents = env.n_teams[1]
        self.switch_side = switch_side
        self.device = device
        self.vectorize = vectorize
        self.discrete_actions = env.discrete_actions

    def apply(self, obs, rng):

        obs = torch.tensor(np.asarray(obs), device=self.device)

        if not self.vectorize:
            obs = torch.unsqueeze(obs, 0)
        obs = torch.moveaxis(obs, -1, 2)
        if self.switch_side:
            obs = obs[:, self.n_ego_agents:, ::]
        else:
            obs = obs[:, :self.n_ego_agents, ::]
        obs = obs.float().contiguous()
        masks = torch.ones(1, 1).to(self.device)
        with torch.no_grad():
            value, action, _, self.recurrent_hidden_states = self.actor_critic.act(
                obs.float(), self.recurrent_hidden_states, masks, deterministic=True)
        if not self.vectorize:
            action = torch.squeeze(action, 0)

        action = action.squeeze(-1)

        if self.discrete_actions:
            action = jnp.array(action.detach().cpu().numpy().astype(np.int32))
        else:
            action = jnp.array(action.detach().cpu().numpy().astype(np.float32))


        if self.switch_side:
            if self.vectorize:
                action = jnp.pad(action, ((0, 0),(self.n_ego_agents,0)), 'constant', constant_values=(0))
            else:
                action = jnp.pad(action, ((self.n_ego_agents,0)), 'constant', constant_values=(0))
        else:
            if self.vectorize:
                action = jnp.pad(action, ((0, 0),(0,self.n_opp_agents)), 'constant', constant_values=(0))
            else:
                action = jnp.pad(action, ((0, self.n_opp_agents)), 'constant', constant_values=(0))

        return action

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
            # CircleRandomPolicy(env),
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
        self.total_games_tie = 0

    @property
    def win_rate_a(self):
        return self.team_a_wins * 100 / self.total_games

    @property
    def win_rate_b(self):
        return self.team_b_wins * 100 / self.total_games

    @property
    def tie_rate(self):
        return (self.total_games - self.team_a_wins - self.team_b_wins) * 100 / self.total_games

    def __str__(self):
        return (
            f"Team A {self.team_a_wins}/{self.total_games-self.total_games_tie} ({self.team_a_wins*100/self.total_games:0.1f}%) wins [{self.team_a_reward/self.total_games:0.1f} mean return]"
            + f", Team B {self.team_b_wins}/{self.total_games} ({self.team_b_wins*100/self.total_games:0.1f}%) wins [{self.team_b_reward/self.total_games:0.1f} mean return]"
            + f"  Tie {self.total_games_tie}/{self.total_games} ({self.total_games_tie*100/self.total_games:0.1f}%) ties"
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
        self.total_games_tie = self.total_games - self.team_a_wins - self.team_b_wins

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




def loop_env(env, policy=None, device="cpu", headless=False, swith_side = False):
    evaluator = Evaluator(env)
    viewer = GigastepViewer(frame_size=84 * 2, show_num_agents=0 if env.discrete_actions else env.n_agents)

    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(5):
            ep_done = False
            key, rng = jax.random.split(rng, 2)
            state, obs = env.reset(key)
            
            ego = Torch_Policy(policy=policy,env=env,device=device,vectorize = False, switch_side = swith_side)
            
            while not ep_done:
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action_ego = ego.apply(obs, key2)

                action_opp = opponent.apply(obs, key2)

                if swith_side:
                    action = evaluator.merge_actions(action_ego, action_opp)
                else:
                    action = evaluator.merge_actions(action_opp, action_ego)

                rng, key = jax.random.split(rng, 2)
                state, obs, r, dones, ep_done = env.step(state, action, key)
                evaluator.update_step(r, dones, ep_done)
                if not headless:
                    img = viewer.draw(env, state, obs)
                    # if viewer.should_pause:
                    #     while True:
                    #         img = viewer.draw(env, state, obs)
                    #         time.sleep(SLEEP_TIME)
                    #         if viewer.should_pause:
                    #             break
                    if viewer.should_quit:
                        sys.exit(1)
                time.sleep(SLEEP_TIME)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)
    return [evaluator.team_a_wins / evaluator.total_games,
            evaluator.team_b_wins / evaluator.total_games,
            evaluator.total_games_tie / evaluator.total_games]


def loop_env_vectorized(env, policy=None, device="cpu", switch_side = False):
    evaluator = Evaluator(env)
    batch_size = 20
    rng = jax.random.PRNGKey(3)

    for opponent in evaluator.policies:
        print("Opponent", str(opponent))
        for ep_idx in range(1):
            ep_done = np.zeros(batch_size, dtype=jnp.bool_)
            key, rng = jax.random.split(rng, 2)
            key = jax.random.split(key, batch_size)
            state, obs = env.v_reset(key)
            t = 0

            ego = Torch_Policy(policy=policy,env=env,device=device,vectorize = True)
            
            while not jnp.all(ep_done):
                rng, key, key2 = jax.random.split(rng, 3)
                if policy is None:
                    action_ego = jnp.zeros(
                        (batch_size, env.n_agents, 3)
                    )  # ego does nothing
                else:
                    action_ego = ego.apply(obs, key2)
                
                key2 = jax.random.split(key2, batch_size)
                action_opp = opponent.v_apply(obs, key2)

                action = evaluator.v_merge_actions(action_ego, action_opp)

                rng, key = jax.random.split(rng, 2)
                key = jax.random.split(key, batch_size)
                state, obs, r, dones, ep_done = env.v_step(state, action, key)
                evaluator.update_step(r, dones, ep_done)

                time.sleep(SLEEP_TIME)
                t += 1
                # print("t", t, "ep_done", ep_done)
            evaluator.update_episode()
            print(str(evaluator))
            # if frame_idx > 400:
            #     sys.exit(1)

    return [evaluator.team_a_wins / evaluator.total_games,
            evaluator.team_b_wins / evaluator.total_games,
            evaluator.total_games_tie / evaluator.total_games]

