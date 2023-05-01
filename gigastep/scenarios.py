from gigastep import GigastepEnv
import jax.numpy as jnp


class ScenarioBuilder:
    def __init__(self):
        self._per_agent_team = []
        self._per_agent_sprites = []
        self._per_agent_max_health = []
        self._per_agent_range = []
        self._per_agent_thrust = []
        self._map = "all"
        self._map_size = None

    def set_map(self, map, map_size=None):
        if map not in ("all", "empty"):
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
        thrust: float = 1,
    ):
        self._per_agent_team.append(team)
        self._per_agent_sprites.append(sprite)
        self._per_agent_max_health.append(max_health)
        self._per_agent_range.append(range)
        self._per_agent_thrust.append(thrust)

    def add_type(self, team, agent_type):
        if agent_type == "default":
            self.add(team=team, sprite=1, max_health=1, range=1, thrust=1)
        elif agent_type == "tank":
            self.add(team=team, sprite=7, max_health=3, range=1, thrust=1)
        elif agent_type == "sniper":
            self.add(team=team, sprite=3, max_health=1, range=2, thrust=1)
        elif agent_type == "scout":
            self.add(team=team, sprite=5, max_health=1, range=1, thrust=2)
        elif agent_type == "boss":
            self.add(team=team, sprite=6, max_health=3, range=2, thrust=1.2)
        else:
            raise ValueError(f"Unknown agent type {agent_type}")

    def make(self, **kwargs):
        """Instantiates the GigastepEnv based on the build environment

        :param kwargs: Named arguments will be passed to the GigastepEnv constructor (__init__)
        :return: A GigastepEnv object
        """
        scenario_args = self.get_kwargs()
        # Overwrite named args that are passed to make
        for k, v in kwargs.items():
            scenario_args[k] = v
        return GigastepEnv(**scenario_args)

    def get_kwargs(self):
        return {
            "n_agents": len(self._per_agent_team),
            "per_agent_team": jnp.array(self._per_agent_team),
            "per_agent_sprites": jnp.array(self._per_agent_sprites),
            "per_agent_max_health": jnp.array(self._per_agent_max_health),
            "per_agent_range": jnp.array(self._per_agent_range),
            "per_agent_thrust": jnp.array(self._per_agent_thrust),
            "maps": self._map,
            "limit_x": self._map_size[0] if self._map_size is not None else None,
            "limit_y": self._map_size[1] if self._map_size is not None else None,
        }

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
        return builder


_builtin_scenarios = {
    "identical_20_vs_20": {
        "team_0": {"default": 20},
        "team_1": {"default": 20},
        "map": "empty",
        "map_size": (20, 20),
    },
    "special_20_vs_20": {
        "team_0": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
        "team_1": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
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
    },
    "identical_5_vs_5": {
        "team_0": {"default": 5},
        "team_1": {"default": 5},
        "map": "empty",
    },
    "special_5_vs_5": {
        "team_0": {"tank": 1, "sniper": 1, "scout": 1, "default": 2},
        "team_1": {"tank": 1, "sniper": 1, "scout": 1, "default": 2},
    },
    "identical_5_vs_1": {
        "team_0": {"default": 5},
        "team_1": {"boss": 1},
    },
    "special_5_vs_1": {
        "team_0": {"tank": 1, "sniper": 1, "scout": 1, "default": 2},
        "team_1": {"boss": 1},
    },
    "identical_10_vs_3": {
        "team_0": {"default": 10},
        "team_1": {"boss": 3},
    },
    "special_10_vs_3": {
        "team_0": {"tank": 3, "sniper": 3, "scout": 3, "default": 1},
        "team_1": {"boss": 3},
    },
    "identical_20_vs_5": {
        "team_0": {"default": 20},
        "team_1": {"boss": 5},
    },
    "special_20_vs_5": {
        "team_0": {"tank": 5, "sniper": 5, "scout": 5, "default": 5},
        "team_1": {"boss": 5},
    },
}


def make_scenario(name, **kwargs):
    """Instantiates a GigastepEnv by the name of the built-in scenarios


    :param name: Name of the scenario. Use ``gigastep.list_scenarios()`` to get a list of all availabel scenarios
    :param kwargs: Named arguments will be passed to the GigastepEnv constructor (__init__)
    :return: A GigastepEnv object
    """
    if name not in _builtin_scenarios.keys():
        raise ValueError(f"Scenario {name} not found.")

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