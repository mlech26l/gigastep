from typing import Any, Dict, List, Optional, Union

import dm_env
import jax
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

import jax.numpy as jnp

from mava import types
from mava.utils.wrapper_utils import (
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class GigastepParallelEnvWrapper(ParallelEnvWrapper):
    """Environment wrapper for Gigastep MARL environments."""

    def __init__(
        self,
        environment,
        random_seed: Optional[int] = 42,
    ):
        """Constructor for parallel Gigastep wrapper.

        Args:
            environment: parallel Gigastep env.
        """
        self._environment = environment
        self._reset_next_step = True

        team_prefix = ("a_", "b_") # NOTE: need to make sure pytree follows the correct order
        team_cnt = [0, 0]
        self._possible_agents = []
        for i, team_id in enumerate(np.asarray(self._environment.teams)):
            agent_id = team_prefix[team_id] + f"{team_cnt[team_id]:03d}"
            self._possible_agents.append(agent_id)
            team_cnt[team_id] += 1
        self._possible_agents_tree = jax.tree_util.tree_structure({k: 0 for k in self.possible_agents})

        float_type = self._environment.observation_space.dtype.type
        int_type = self._environment.action_space.dtype.dtype

        obs_space = spaces.Box(
            low=np.array(self._environment.observation_space.low, dtype=float_type),
            high=np.array(self._environment.observation_space.high, dtype=float_type),
            shape=self._environment.observation_space.shape,
            dtype=float_type,
        )
        observation = _convert_to_spec(obs_space)
        if self._environment.discrete_actions:
            act_space = spaces.Discrete(n=self._environment.action_space.n)
            # legal action mask should be a vector of ones and zeros
            legal_actions = specs.BoundedArray(
                shape=(act_space.n,),
                dtype=act_space.dtype,
                minimum=np.zeros(act_space.shape),
                maximum=np.zeros(act_space.shape) + 1,
                name=None,
            )
        else:
            act_space = spaces.Box(
                low=np.array(self._environment.action_space.low),
                high=np.array(self._environment.action_space.high),
                shape=self._environment.action_space.shape,
                dtype=int_type,
            )
            legal_actions = _convert_to_spec(act_space)

        observation_specs = {}
        for agent in self.possible_agents:
            observation_specs[agent] = types.OLT(
                observation=observation,
                legal_actions=legal_actions,
                terminal=specs.Array((1,), np.float32),
            )
        self._observation_specs = observation_specs

        action_specs = {}
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(act_space)
        self._action_specs = action_specs

        self._rng = jax.random.PRNGKey(random_seed)
        self._key_reset = None
        self._key_step = None
        self._env_state = None
        self._ep_dones = False # env-level done

    def _jarr_to_dict(self, jarr, to_numpy=False):
        if to_numpy:
            out = np.asarray(jarr)
        else:
            out = jarr
        out = jax.tree_util.tree_unflatten(self._possible_agents_tree, out)
        return out

    def _env_reset(self):
        self._rng, self._key_reset = jax.random.split(self._rng, 2)
        self._env_state, observe = self._environment.reset(self._key_reset)
        observe = self._jarr_to_dict(observe)
        return observe
    
    def _env_step(self, actions):
        self._rng, self._key_step = jax.random.split(self._rng, 2)
        self._env_state, obs, rewards, dones, self._ep_dones = self._environment.step(self._env_state, actions, self._key_step)
        obs = self._jarr_to_dict(obs)
        rewards = self._jarr_to_dict(rewards)
        dones = self._jarr_to_dict(dones)
        return obs, rewards, dones, {}

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        observe = self._env_reset()

        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }

        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        return parameterized_restart(rewards, self._discounts, observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """
        if self._reset_next_step:
            return self.reset()

        # TODO: append for all possible agents
        actions = jnp.array(jax.tree_util.tree_flatten(actions)[0])

        observations, rewards, dones, infos = self._env_step(actions)

        rewards = self._convert_reward(rewards)
        observations = self._convert_observations(observations, dones)

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
            # Terminal discount should be 0.0 as per dm_env
            discount = {
                agent: convert_np_type(self.discount_spec()[agent].dtype, 0.0)
                for agent in self.possible_agents
            }
        else:
            self._step_type = dm_env.StepType.MID
            discount = self._discounts

        timestep = dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=discount,
            step_type=self._step_type,
        )

        return timestep

    def extras_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        return {}

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._ep_dones

    def _convert_reward(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Convert rewards to be dm_env compatible.

        Args:
            rewards (Dict[str, float]): rewards per agent.
        """
        rewards_spec = self.reward_spec()
        rewards_return = {}
        for agent in self.possible_agents:
            if agent in rewards:
                rewards_return[agent] = convert_np_type(
                    rewards_spec[agent].dtype, rewards[agent]
                )
            # Default reward
            else:
                rewards_return[agent] = convert_np_type(rewards_spec[agent].dtype, 0)
        return rewards_return

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        """Convert PettingZoo observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        return convert_dm_compatible_observations(
            observes,
            dones,
            self.observation_spec(),
            self.env_done(),
            self.possible_agents,
        )

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        return self._observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]: spec for actions.
        """
        return self._action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def get_state(self) -> Optional[Dict]:
        """Retrieve state from environment.

        Returns:
            environment state.
        """
        state = None
        return state

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        if hasattr(self._environment, "get_stats"):
            return self._environment.get_stats()
        elif (
            hasattr(self._environment, "unwrapped")
            and hasattr(self._environment.unwrapped, "env")
            and hasattr(self._environment.unwrapped.env, "get_stats")
        ):
            return self._environment.unwrapped.env.get_stats()
        else:
            return None

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._environment.agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._possible_agents

    @property
    def environment(self):
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    @property
    def current_agent(self) -> Any:
        """Current active agent.

        Returns:
            Any: current agent.
        """
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
