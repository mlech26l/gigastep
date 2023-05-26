from typing import Tuple, Optional
import argparse
import numpy as np
from gym.spaces import Dict as GymDict, Box

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import EnvID, MultiAgentDict, MultiEnvDict

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

import jax
# jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
from gigastep import make_scenario


class NotUsingVMapError(Exception):
    def __init__(self, message=None):
        if message is None:
            message = "Please use vmap implementation instead"
        self.message = message
        super().__init__(self.message)


class Gigastep(MultiAgentEnv):
    DEFAULT_POLICY_MAPPING_DICT = {
        "description": "default configuration of gigastep",
        "team_prefix": ("red_", "blue_"),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    }
    EPISODE_LIMIT = 500

    def __init__(self, env_config):
        print("================================instantiate================================")
        map = env_config.pop("map_name", None)
        env_config["obs_type"] = env_config.pop("obs_type", "vector")
        seed = env_config.pop("seed", 42)

        self.env = make_scenario(map, **env_config)
        self.action_space = Box(
            low=np.array(self.env.action_space.low),
            high=np.array(self.env.action_space.high),
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype.type,
        )
        self.observation_space = GymDict({"obs": Box(
            low=np.array(self.env.observation_space.low),
            high=np.array(self.env.observation_space.high),
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype.type)})
        
        env_config["map_name"] = map
        env_config["seed"] = seed
        self.env_config = env_config

        team_prefix = self.DEFAULT_POLICY_MAPPING_DICT["team_prefix"]
        team_cnt = [0, 0]
        self.agents = []
        for i, team_id in enumerate(np.asarray(self.env.teams)):
            agent_id = team_prefix[team_id] + f"{team_cnt[team_id]}"
            self.agents.append(agent_id)
            team_cnt[team_id] += 1
        self.num_agents = self.env.n_agents

        self.initialize = False

    def set_num_envs(self, num_envs):
        self.num_envs = num_envs

    def reset(self):
        raise NotUsingVMapError

    def step(self, action_dict):
        raise NotUsingVMapError
    
    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict,
                            MultiEnvDict, MultiEnvDict]:
        if not self.initialize:
            self.rng = jax.random.PRNGKey(self.env_config["seed"])
            self.rng, self.key_reset = jax.random.split(self.rng, 2)
            self.key_reset = jax.random.split(self.key_reset, self.num_envs)
            self.key_step = None

            self.ep_dones = jnp.zeros(self.num_envs, dtype=jnp.bool_) # env-level done
            self.last_state, self.last_obs = self.env.v_reset(self.key_reset)
            self.last_rewards = jnp.zeros((self.num_envs, self.num_agents), dtype=jnp.float32)
            self.last_dones = jnp.zeros((self.num_envs, self.num_agents), dtype=jnp.bool_)

            self.initialize = True

        observations = self._jarr_to_medict(self.last_obs, key_name="obs")
        rewards = self._jarr_to_medict(self.last_rewards)
        # dones = self._jarr_to_medict(self.last_dones)
        # for i, v in enumerate(self._to_njx(self.ep_dones)):
        #     dones[i]["__all__"] = v
        dones = dict()
        for i, v in enumerate(self._to_njx(self.ep_dones)):
            dones[i] = {"__all__": v}

        infos = {env_id: {} for env_id in range(self.num_envs)}

        return observations, rewards, dones, infos, {}

    def send_actions(self, action_dict: MultiEnvDict) -> None:
        default_val_fn = lambda: np.zeros(self.env.action_space.shape, dtype=np.float32)
        action = self._medict_to_jarr(action_dict, default_val_fn)

        self.rng, self.key_step = jax.random.split(self.rng, 2)
        self.key_step = jax.random.split(self.key_step, self.num_envs)
        self.last_state, self.last_obs, self.last_rewards, self.last_dones, self.ep_dones \
            = self.env.v_step(self.last_state, action, self.key_step)

    def try_reset(
        self,
        env_id: Optional[EnvID] = None
    ) -> Optional[MultiAgentDict]:
        if self.ep_dones[env_id].item():
            # TODO: non-elegant way to step one env only
            self.rng, self.key_reset = jax.random.split(self.rng, 2)
            ep_dones_i = jnp.zeros(self.num_envs, dtype=jnp.bool_)
            ep_dones_i = ep_dones_i.at[env_id].set(True)
            self.last_state, self.last_obs = self.env.reset_done_episodes(
                self.last_state, self.last_obs, ep_dones_i, self.key_reset)
            self.ep_dones = self.ep_dones.at[env_id].set(False)
            
            # TODO: not a good practice of converting data of all envs and only extract one of them
            obs = self._jarr_to_medict(self.last_obs, key_name="obs")[env_id]
        else:
            obs = None

        return obs
    
    def close(self):
        pass

    def render(self, mode=None):
        self.env.render()
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.EPISODE_LIMIT,
            "policy_mapping_info": {self.env_config["map_name"]: self.DEFAULT_POLICY_MAPPING_DICT},
        }
        return env_info

    def _jarr_to_medict(self, data, key_name=None, skip_by_done=False) -> MultiEnvDict:
        data_njx = self._to_njx(data)

        if skip_by_done:
            last_dones_njx = self._to_njx(self.last_dones)

        multi_env_dict = dict()
        for env_idx, env_id in enumerate(range(self.num_envs)):
            multi_env_dict[env_id] = dict()
            for agent_idx, agent_id in enumerate(self.agents):
                if not skip_by_done or not last_dones_njx[env_id, agent_idx]:
                    datum = data_njx[env_idx, agent_idx]
                    if key_name is not None:
                        datum = {key_name: datum}
                    multi_env_dict[env_id][agent_id] = datum

        return multi_env_dict
    
    def _medict_to_jarr(self, data, default_val_fn=None):
        data_jx = []
        for env_id in range(self.num_envs):
            data_jx.append([])
            for agent_id in self.agents:
                if agent_id not in data[env_id].keys() and default_val_fn is not None:
                    datum_jx = default_val_fn()
                else:
                    datum_jx = data[env_id][agent_id]
                data_jx[-1].append(datum_jx)
        data_jx = jnp.asarray(data_jx)

        return data_jx
    
    def _to_njx(self, data):
        return np.asarray(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="mappo",
                        choices=["mappo", "ippo", "matrpo", "maa2c", "itrpo", "ia2c",
                                 "happo", "hatrpo", "vdppo", "vda2c"])
    parser.add_argument("--num-workers", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timesteps-total", type=int, default=1e7)
    parser.add_argument("--local-mode", action="store_true", help="Set this if using GPU in JAX")
    parser.add_argument("--obs-type", type=str, default="vector",
                        choices=["rgb", "vector"])
    parser.add_argument("--map-name", type=str, default="identical_5_vs_5",
                        choices=["identical_5_vs_5"])
    parser.add_argument("--hyperparam-source", type=str, default="test")
    args = parser.parse_args()

    ENV_REGISTRY["gigastep"] = Gigastep
    COOP_ENV_REGISTRY["gigastep"] = Gigastep

    env = marl.make_env(
        environment_name="gigastep",
        map_name=args.map_name,
        obs_type=args.obs_type,
        seed=args.seed,
    )

    alg_cls = getattr(marl.algos, args.alg)
    alg = alg_cls(hyperparam_source=args.hyperparam_source)

    model = marl.build_model(env, alg, {"core_arch": "gru", "hidden_state_size": 128})

    alg.fit(
        env, 
        model,
        stop={'episode_reward_mean': 2000, 'timesteps_total': args.timesteps_total}, 
        local_mode=args.local_mode, 
        num_gpus=1,
        num_workers=args.num_workers,
        num_envs_per_worker=args.num_envs,
        share_policy='all',
        checkpoint_freq=50,
    )
