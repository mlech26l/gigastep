import copy

from gigastep import GigastepEnv
import jax.numpy as jnp


class ScenarioBuilder:
    def __init__(self):
        self._per_agent_team = []
        self._per_agent_sprites = []
        self._per_agent_max_health = []
        self._per_agent_range = []
        self._per_agent_damage_range = []
        self._per_agent_thrust = []
        self._per_agent_idle_reward = []
        self._map = "all"
        self._map_size = None
        self._kwargs = {}

    def add_kwarg(self, k, v):
        self._kwargs[k] = v

    def set_map(self, map, map_size=None):
        if map not in ("all", "empty", "two_rooms1", "four_rooms", "center_block"):
            raise ValueError(f"Unknown map {map}")
        self._map = map
        if map_size is not None:
            self._map_size = map_size

    def add(
        self,
        team: int = 0,
        sprite: int = 1,
        max_health: float = 1,
        range: float = 1,
        damage_range: float = 1,
        idle_reward: float = 0,
        thrust: float = 1,
    ):
        self._per_agent_team.append(team)
        self._per_agent_sprites.append(sprite)
        self._per_agent_max_health.append(max_health)
        self._per_agent_range.append(range)
        self._per_agent_damage_range.append(damage_range)
        self._per_agent_thrust.append(thrust)
        self._per_agent_idle_reward.append(idle_reward)

    def add_type(self, team, agent_type):
        if agent_type == "default":
            self.add(team=team, sprite=1, max_health=1, range=1, thrust=1)
        elif agent_type == "tank":
            self.add(team=team, sprite=7, max_health=3, range=1, thrust=1)
        elif agent_type == "sniper":
            self.add(team=team, sprite=3, max_health=0.5, range=2, thrust=1)
        elif agent_type == "scout":
            self.add(team=team, sprite=5, max_health=1, range=1, thrust=2)
        elif agent_type == "boss":
            self.add(team=team, sprite=6, max_health=3, range=1, thrust=0.8)
        elif agent_type == "seeker":
            self.add(team=team)
        elif agent_type == "hider":
            self.add(team=team, damage_range=0, idle_reward=1)
        else:
            raise ValueError(f"Unknown agent type {agent_type}")

    def make(self, **kwargs):
        """Instantiates the GigastepEnv based on the build environment

        :param kwargs: Named arguments will be passed to the GigastepEnv constructor (__init__)
        :return: A GigastepEnv object
        """
        scenario_args = self.get_kwargs()
        for k, v in self._kwargs.items():
            scenario_args[k] = v

        # Overwrite named args that are passed to make
        for k, v in kwargs.items():
            scenario_args[k] = v
        return GigastepEnv(**scenario_args)

    def get_kwargs(self):
        kwargs = {
            "n_agents": len(self._per_agent_team),
            "per_agent_team": jnp.array(self._per_agent_team),
            "per_agent_sprites": jnp.array(self._per_agent_sprites),
            "per_agent_max_health": jnp.array(self._per_agent_max_health),
            "per_agent_range": jnp.array(self._per_agent_range),
            "per_agent_damage_range": jnp.array(self._per_agent_damage_range),
            "per_agent_thrust": jnp.array(self._per_agent_thrust),
            "per_agent_idle_reward": jnp.array(self._per_agent_idle_reward),
            "maps": self._map,
        }
        if self._map_size is not None:
            kwargs["limit_x"] = self._map_size[0]
            kwargs["limit_y"] = self._map_size[1]
        return kwargs

    @classmethod
    def from_config(cls, config):
        builder = cls()
        for agents_type, num in config["team_0"].items():
            for _ in range(num):
                builder.add_type(0, agents_type)
        for agents_type, num in config["team_1"].items():
            for _ in range(num):
                builder.add_type(1, agents_type)

        map = config.get("map", "all")
        map_size = config.get("map_size", None)
        builder.set_map(map, map_size)
        kwargs = config.get("kwargs", {})
        for k, v in kwargs.items():
            builder.add_kwarg(k, v)
        return builder


_builtin_scenarios = {
    "hide_and_seek_5_vs_5": {
        "team_0": {"seeker": 5},
        "team_1": {"hider": 5},
        "map": "empty",
        "kwargs": {
            "damage_per_second": 10,
            "damage_cone_depth": 1.0,
            "damage_cone_angle": jnp.pi,  # +-, thus 360 degrees
            "collision_range": 0.0,  # no collision
            "max_episode_length": 500,
            "reward_game_won": 50,
            "reward_defeat_one_opponent": 5,
            "reward_detection": 0,
            "reward_damage": 10,
            "reward_idle": 0,
            "reward_agent_disabled": 10,
            "reward_collision_agent": 0,
            "reward_collision_obstacle": 10,
        },
    },
    "waypoint_5_vs_5": {
        "team_0": {"default": 5},
        "team_1": {"default": 5},
        "map": "empty",
        "kwargs": {
            "damage_cone_depth": 0.0,
            "episode_ends_one_team_dead": False,
            "max_episode_length": 500,
            "enable_waypoints": True,
            "reward_game_won": 0,
            "reward_defeat_one_opponent": 0,
            "reward_detection": 0,
            "reward_damage": 0,
            "reward_idle": 0,
            "reward_agent_disabled": 0,
            "reward_collision_agent": 0,
            "reward_collision_obstacle": 0,
            "reward_hit_waypoint": 50,
        },
    },
    "identical_20_vs_20": {
        "team_0": {"default": 20},
        "team_1": {"default": 20},
        "map": "empty",
        "map_size": (20, 20),
    },
    "special_20_vs_20": {
        "team_0": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
        "team_1": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
        "map": "empty",
        "map_size": (20, 20),
    },
    "identical_10_vs_10": {
        "team_0": {"default": 10},
        "team_1": {"default": 10},
        "map": "empty",
    },
    "special_10_vs_10": {
        "team_0": {"tank": 3, "sniper": 3, "scout": 3, "default": 1},
        "team_1": {"tank": 3, "sniper": 3, "scout": 3, "default": 1},
        "map": "empty",
    },
    "identical_5_vs_5": {
        "team_0": {"default": 5},
        "team_1": {"default": 5},
        "map": "empty",
    },
    "special_5_vs_5": {
        "team_0": {"tank": 1, "sniper": 1, "scout": 1, "boss": 1, "default": 1},
        "team_1": {"tank": 1, "sniper": 1, "scout": 1, "boss": 1, "default": 1},
        "map": "empty",
    },
    "identical_1_vs_1": {
        "team_0": {"default": 1},
        "team_1": {"default": 1},
        "map": "empty",
    },
    "identical_2_vs_2": {
        "team_0": {"default": 2},
        "team_1": {"default": 2},
        "map": "empty",
    },
    "identical_5_vs_1": {
        "team_0": {"default": 5},
        "team_1": {"boss": 1},
        "map": "empty",
    },
    "special_5_vs_1": {
        "team_0": {"tank": 1, "sniper": 1, "scout": 1, "default": 2},
        "team_1": {"boss": 1},
        "map": "empty",
    },
    "identical_10_vs_3": {
        "team_0": {"default": 10},
        "team_1": {"boss": 3},
        "map": "empty",
    },
    "special_10_vs_3": {
        "team_0": {"tank": 3, "sniper": 3, "scout": 3, "default": 1},
        "team_1": {"boss": 3},
        "map": "empty",
    },
    "identical_20_vs_5": {
        "team_0": {"default": 20},
        "team_1": {"boss": 5},
        "map": "empty",
    },
    "special_20_vs_5": {
        "team_0": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
        "team_1": {"boss": 5},
        "map": "empty",
    },
    "identical_20_vs_20_center_block": {
        "team_0": {"default": 20},
        "team_1": {"default": 20},
        "map": "center_block",
        "map_size": (20, 20),
    },
    "identical_20_vs_20_two_rooms1": {
        "team_0": {"default": 20},
        "team_1": {"default": 20},
        "map": "two_rooms1",
        "map_size": (20, 20),
    },
    "identical_10_vs_10_center_block": {
        "team_0": {"default": 10},
        "team_1": {"default": 10},
        "map": "center_block",
    },
    "identical_10_vs_10_two_rooms1": {
        "team_0": {"default": 10},
        "team_1": {"default": 10},
        "map": "two_rooms1",
    },
    "identical_5_vs_5_center_block": {
        "team_0": {"default": 5},
        "team_1": {"default": 5},
        "map": "center_block",
    },
    "identical_2_vs_2_center_block": {
        "team_0": {"default": 2},
        "team_1": {"default": 2},
        "map": "center_block",
    },
    "identical_5_vs_5_two_rooms1": {
        "team_0": {"default": 5},
        "team_1": {"default": 5},
        "map": "two_rooms1",
    },
    # # Large scale scenarios
    # "identical_50_vs_50": {
    #     "team_0": {"default": 50},
    #     "team_1": {"default": 50},
    #     "map": "empty",
    #     "map_size": (40, 40),
    # },
    # "identical_100_vs_100": {
    #     "team_0": {"default": 100},
    #     "team_1": {"default": 100},
    #     "map": "empty",
    #     "map_size": (100, 100),
    # },
    # "identical_1000_vs_1000": {
    #     "team_0": {"default": 1000},
    #     "team_1": {"default": 1000},
    #     "map": "empty",
    #     "map_size": (1000, 1000),
    # },
    # "identical_10000_vs_10000": {
    #     "team_0": {"default": 10000},
    #     "team_1": {"default": 10000},
    #     "map": "empty",
    #     "map_size": (10000, 10000),
    # },
}


def _make_deterministic_variants(list_of_scenarios):
    for name in list_of_scenarios:
        if name.endswith("_det"):
            continue
        det_name = name + "_det"
        if det_name in _builtin_scenarios.keys():
            continue
            # already exists
        deterministic_scenario = copy.deepcopy(_builtin_scenarios[name])
        if "kwargs" not in deterministic_scenario:
            deterministic_scenario["kwargs"] = {}
        deterministic_scenario["kwargs"]["use_stochastic_obs"] = False
        deterministic_scenario["kwargs"]["use_stochastic_comm"] = False
        deterministic_scenario["kwargs"]["cone_depth"] = 100
        deterministic_scenario["kwargs"]["cone_angle"] = 2 * jnp.pi
        deterministic_scenario["kwargs"]["reward_detection"] = 0

        _builtin_scenarios[det_name] = deterministic_scenario


_make_deterministic_variants(list(_builtin_scenarios.keys()))


def make_scenario(name, **kwargs):
    """Instantiates a GigastepEnv by the name of the built-in scenarios


    :param name: Name of the scenario. Use ``gigastep.list_scenarios()`` to get a list of all availabel scenarios
    :param kwargs: Named arguments will be passed to the GigastepEnv constructor (__init__)
    :return: A GigastepEnv object
    """
    if name not in _builtin_scenarios.keys():
        raise ValueError(f"Scenario {name} not found.")

    warning_already_printed = False
    if "use_stochastic_obs" in kwargs and kwargs["use_stochastic_obs"] is False:
        print(
            "WARNING: You are overriding the use_stochastic_obs flag."
            "If you want to use a fully observable scenario, "
            f"use the _det variant of the scenario: make_scenario('{name}_det')",
        )
        warning_already_printed = True
    if "use_stochastic_comm" in kwargs and kwargs["use_stochastic_comm"] is False:
        if not warning_already_printed:
            print(
                "WARNING: You are overriding the use_stochastic_comm flag."
                "If you want to use a fully observable scenario, "
                f"use the _det variant of the scenario: make_scenario('{name}_det')",
            )
        warning_already_printed = True
    if "cone_depth" in kwargs:
        if not warning_already_printed:
            print(
                "WARNING: You are overriding the cone_depth value."
                "If you want to use a fully observable scenario, "
                f"use the _det variant of the scenario: make_scenario('{name}_det')",
            )
        warning_already_printed = True
    if "cone_angle" in kwargs:
        if not warning_already_printed:
            print(
                "WARNING: You are overriding the cone_angle value."
                "If you want to use a fully observable scenario, "
                f"use the _det variant of the scenario: make_scenario('{name}_det')",
            )
        warning_already_printed = True
    scenario = _builtin_scenarios[name]
    scenario = ScenarioBuilder.from_config(scenario)
    return scenario.make(**kwargs)


def list_scenarios():
    return list(_builtin_scenarios.keys())


if __name__ == "__main__":
    print("| Scenario | Map |")
    print("| --- | --- |")
    for scenario in list_scenarios():
        print(f"| {scenario} |   |")