import jax
import jax.numpy as jnp
import torch
import sys
import time
import os
import bisect


import numpy as np

from gigastep import make_scenario, GigastepViewer


SLEEP_TIME = 0.01


class EvaluatiorPolicy:
    def apply(self, obs, rng):
        raise NotImplementedError()


class Torch_Policy(EvaluatiorPolicy):
    def __init__(self, env, policy, device = "cpu", vectorize = False, switch_side = False):

        self.actor_critic = policy
        if vectorize:
            self.recurrent_hidden_states = (
                torch.zeros(
                    env.n_agents, policy.recurrent_hidden_state_size, device=device
                )
                if policy is not None
                else None
            )
        else:
            self.recurrent_hidden_states = (
                torch.zeros(
                    env.n_agents, policy.recurrent_hidden_state_size, device=device
                )
                if policy is not None
                else None
            )
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
                obs.float(), self.recurrent_hidden_states, masks, deterministic=True
            )
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




class PolicyPoolJax(EvaluatiorPolicy):
    def __int__(self, env, ActorCriticMLP, ckpt,
                config = {"ACTIVATION": "relu",
                          "ENV_CONFIG": {"obs_type": "vector"}
                          }
                ):

        from functools import partial
        from flax.training import orbax_utils
        import orbax.checkpoint

        forbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        train_state, _ =  forbax_checkpointer.restore(ckpt)

        make_network = partial(ActorCriticMLP,
                               env.action_space.n,
                               activation=config["ACTIVATION"],
                               teams=env.teams,
                               has_cnn=config["ENV_CONFIG"]["obs_type"] == "rgb" and config["USE_CNN"],
                               obs_shape=env.observation_space.shape)
        network = make_network()
        def action_fn_base(network_params, obs, rng):
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(network_params, obs)
            action = pi.sample(seed=_rng)
            return action

        self.action_fn = partial(action_fn_base, train_state.params)

    def apply(self, obs, rng):
        action = self.action_fn(obs, rng)
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
            self.action1 = jnp.argmin(
                jnp.linalg.norm(self.env.action_lut - jnp.array([[1, 0, 0]]), axis=1)
            )
            self.action2 = jnp.argmin(
                jnp.linalg.norm(self.env.action_lut - jnp.array([[-1, 0, 0]]), axis=1)
            )
        self.v_apply = jax.vmap(self.apply, in_axes=(0, 0))

    def apply(self, obs, rng):
        n_agents = obs.shape[0]
        if self.env.discrete_actions:
            directions = jax.random.randint(
                jax.random.PRNGKey(1), shape=(n_agents,), minval=0, maxval=2
            )
            actions = jnp.where(directions == 1, self.action1, self.action2)
        else:
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
        self.total_games_tie = 0

    @property
    def win_rate_a(self):
        return self.team_a_wins * 100 / self.total_games

    @property
    def win_rate_b(self):
        return self.team_b_wins * 100 / self.total_games

    @property
    def tie_rate(self):
        return (
            (self.total_games - self.team_a_wins - self.team_b_wins)
            * 100
            / self.total_games
        )

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



class Rating():
    def __init__(self, device, save_path, env_name, env_cfg,
                 num_match=5):
        self.device = device
        self.save_path = save_path
        self.env_name = env_name
        self.env_cfg = env_cfg
        self.num_match = num_match
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if os.path.isfile(self.save_path + "rating_log.pt"):
            data = torch.load(self.save_path + "rating_log.pt")
            self.rating = data["rating"]
            self.path_pool = data["path_pool"]
            print("rating log loaded")
        else:
            self.rating = []
            self.path_pool = []
            print("failed to load rating log")

    def update_pool_rating(self):
        pool_size = len(self.rating)
        print("Updating pool")
        for i in range(pool_size):
            for j in range(pool_size):
                if i != j:
                    policy_0, _ = torch.load(self.path_pool[i])
                    policy_1, _ = torch.load(self.path_pool[j])
                    self.rating[i], self.rating[j], _ = self._match(
                        policy_0,
                        policy_1,
                        self.rating[i],
                        self.rating[j]
                    )
        print("pool updating is done")
        print(f"rating list is {self.rating}")
        self._save_pool()

    def rate_policy(self, policy, rating_0=1200, add_policy_to_pool=False):

        policy.to(self.device)
        # rating_0 = 1200
        if len(self.rating) == 0:
            policy_1 = None
            rating_0, _, winning_0 = self._match(policy, policy_1,
                                                 rating_0, 0)
        else:
            for i in range(len(self.rating)):
                policy_1, _ = torch.load(self.path_pool[i])
                rating_0, _, _ = self._match(policy, policy_1,
                                             rating_0, self.rating[i])

        rank = bisect.bisect_left(sorted(self.rating), rating_0) + 1

        if len(self.rating) == 0:
            if winning_0 == 1 or add_policy_to_pool:
                position = len(self.rating)
                path = f"pool_{position:04}.pt"
                torch.save([policy, None], f"{self.save_path}{path}")

                self.rating.append(rating_0)
                self.path_pool.append(self.save_path + path)
                self._save_pool()
                print(f"Rating for this agent is done, rating of agent is {rating_0}")
            else:
                print(f"rating of agent is {rating_0}, it is worse than base")

        else:
            if rating_0 > sorted(self.rating)[-1] or add_policy_to_pool:
                position = len(self.rating)
                self.rating.append(rating_0)
                path = f"pool_{position:04}.pt"
                torch.save([policy, None], f"{self.save_path}{path}")
                self.path_pool.append(self.save_path + path)
                self._save_pool()
            else:
                print(f"rating of agent is {rating_0}" +
                      f"which is lower than highest{sorted(self.rating)[-1]}" +
                      f" ranked in {rank}")

        return rating_0

    def _save_pool(self):
        data = {
            "rating": self.rating,
            "path_pool": self.path_pool
        }
        torch.save(data, self.save_path + "rating_log.pt")

    def _match(self, policy_0, policy_1, rate_0, rate_1):
        torch_jax_policy = True
        if torch_jax_policy:
            from enjoy_policy_discrete import evaluation_jax
            policy_0.to(self.device)
            policy_1.to(self.device)

            videos, winning_0 = evaluation_jax(self.env_name, obs_type="rgb",
                                           discrete_actions=True,
                                           actor_critic=policy_0,
                                           actor_critic_opponent=policy_1,
                                           num_of_evaluation=self.num_match,
                                           device=self.device,
                                           headless=True,
                                           env_cfg=self.env_cfg,
                                           show_num_agents=0)
        else:
            env = make_scenario(self.env_name,
                            **vars(self.env_cfg)
                            )
            winning_0, wining_1,tie_rate = loop_env_vectorized_two_Policy(
                        env,
                        policy_ego = None,
                        policy_opp = None,
                        device="cpu",
                        switch_side = False)

        winning_0 = 1 if winning_0 > 0.5 else 0

        Exp_0 = 1 / (1 + 10 ** ((rate_1 - rate_0)/400))
        Exp_1 = 1 / (1 + 10 ** ((rate_0 - rate_1)/400))

        K = 32
        rate_0 = rate_0 + K * (winning_0 - Exp_0)
        rate_1 = rate_1 + K * (1 - winning_0 - Exp_1)

        return rate_0, rate_1, winning_0




def loop_env(env, policy=None, device="cpu", headless=False, swith_side = False):

    evaluator = Evaluator(env)
    viewer = GigastepViewer(
        frame_size=84 * 2, show_num_agents=0 if env.discrete_actions else env.n_agents
    )

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
                    action_ego = jnp.zeros((env.n_agents, 3))  # ego does nothing
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

            ego = Torch_Policy(policy=policy, env=env, device=device, vectorize=True)

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



def loop_env_vectorized_two_Policy(env, policy_ego = None, policy_opp = None, device="cpu", switch_side = False):
    evaluator = Evaluator(env)
    batch_size = 20
    rng = jax.random.PRNGKey(3)

    print("Opponent", str(policy_opp))
    for ep_idx in range(1):
        ep_done = np.zeros(batch_size, dtype=jnp.bool_)
        key, rng = jax.random.split(rng, 2)
        key = jax.random.split(key, batch_size)
        state, obs = env.v_reset(key)
        t = 0

        # Torch policy
        # ego = Torch_Policy(policy=policy, env=env, device=device, vectorize=True)
        # Jax based  
        ego = policy_ego

        while not jnp.all(ep_done):
            rng, key, key2 = jax.random.split(rng, 3)
            if policy_ego  is None:
                action_ego = jnp.zeros(
                    (batch_size, env.n_agents, 3)
                )  # ego does nothing
            else:
                action_ego = ego.apply(obs, key2)

            key2 = jax.random.split(key2, batch_size)
            action_opp = policy_opp.v_apply(obs, key2)

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


