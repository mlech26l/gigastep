from gigastep import GigastepEnv
import jax.numpy as jnp

class ScenarioBuilder:

    def __init__(self):
        self._per_agent_team = []
        self._per_agent_sprites = []
        self._per_agent_max_health = []
        self._per_agent_range = []
        self._per_agent_thrust = []

    def add(self,team:int=0, sprite:int=1, max_health:float=1, range:float=1, thrust:float=1):
        self._per_agent_team.append(team)
        self._per_agent_sprites.append(sprite)
        self._per_agent_max_health.append(max_health)
        self._per_agent_range.append(range)
        self._per_agent_thrust.append(thrust)

    def add_type(self,team, agent_type):
        if agent_type == "default":
            self.add(team=team, sprite=1, max_health=1, range=1, thrust=1)
        elif agent_type == "tank":
            self.add(team=team, sprite=7, max_health=3, range=1, thrust=1)
        elif agent_type == "sniper":
            self.add(team=team, sprite=3, max_health=1, range=2, thrust=1)
        elif agent_type == "scout":
            self.add(team=team, sprite=5, max_health=1, range=1, thrust=2)
        elif agent_type=="boss":
            self.add(team=team, sprite=6, max_health=3, range=2, thrust=1.2)
        else:
            raise ValueError(f"Unknown agent type {agent_type}")

    def make(self):
        GigastepEnv(**self.get_kwargs())

    def get_kwargs(self):
        return {
            "n_agents": len(self._per_agent_team),
            "per_agent_team": jnp.array(self._per_agent_team),
            "per_agent_sprites": jnp.array(self._per_agent_sprites),
            "per_agent_max_health": jnp.array(self._per_agent_max_health),
            "per_agent_range": jnp.array(self._per_agent_range),
            "per_agent_thrust": jnp.array(self._per_agent_thrust),
        }

    @classmethod
    def from_config(cls, config):
        builder = cls()
        for agents in config["team_0"]:
            for k,v in agents.items():
                for _ in range(v):
                    builder.add_type(0, k)
        for agents in config["team_1"]:
            for k,v in agents.items():
                for _ in range(v):
                    builder.add_type(1, k)
        return builder

_builtin_scenarios = {
    "20_vs_20": {
        "team_0": {"default":20},
        "team_1": {"default":20},
    },
}

def get_scenario(name):
    if name not in _builtin_scenarios.keys():
        raise ValueError(f"Scenario {name} not found.")

    scenario = _builtin_scenarios[name]
    if isinstance(scenario, dict):
        return ScenarioBuilder.from_config(scenario).make()
    return _builtin_scenarios[name]()