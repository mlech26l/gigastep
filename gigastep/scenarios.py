from gigastep import GigastepEnv
import jax.numpy as jnp

class ScenarioBuilder:

    def __init__(self):
        self._per_agent_team = []
        self._per_agent_sprites = []
        self._per_agent_max_health = []
        self._per_agent_range = []
        self._per_agent_thrust = []

    def add(self,team:int=0, sprites:int=1, max_health:float=1, range:float=1, thrust:float=1):
        self._per_agent_team.append(team)
        self._per_agent_sprites.append(sprites)
        self._per_agent_max_health.append(max_health)
        self._per_agent_range.append(range)
        self._per_agent_thrust.append(thrust)

    def add_tank_type(self,team=0):
        self.add(team=team, sprites=7, max_health=3, range=1, thrust=1)

    def add_sniper_type(self,team=0):
        self.add(team=team, sprites=3, max_health=1, range=2, thrust=1)

    def add_ranger_type(self,team=0):
        self.add(team=team, sprites=5, max_health=1, range=0, thrust=2)

    def add_default_type(self,team=0):
        self.add(team=team, sprites=1, max_health=1, range=1, thrust=1)

    def add_special_type(self,team=0):
        self.add(team=team, sprites=6, max_health=3, range=2, thrust=1.2)



    def get_kwargs(self):
        return {
            "n_agents": len(self._per_agent_team),
            "per_agent_team": jnp.array(self._per_agent_team),
            "per_agent_sprites": jnp.array(self._per_agent_sprites),
            "per_agent_max_health": jnp.array(self._per_agent_max_health),
            "per_agent_range": jnp.array(self._per_agent_range),
            "per_agent_thrust": jnp.array(self._per_agent_thrust),
        }

def get_5v5_env():
    builder = ScenarioBuilder()
    # 2 default, 1 tank, 1 sniper, 1 ranger
    builder.add_default_type(0)
    builder.add_default_type(0)
    builder.add_tank_type(0)
    builder.add_sniper_type(0)
    builder.add_ranger_type(0)

    builder.add_default_type(1)
    builder.add_default_type(1)
    builder.add_tank_type(1)
    builder.add_sniper_type(1)
    builder.add_ranger_type(1)

    return GigastepEnv(**builder.get_kwargs())

def get_3v3_env():
    builder = ScenarioBuilder()
    # 1 tank, 1 sniper, 1 ranger
    builder.add_tank_type(0)
    builder.add_sniper_type(0)
    builder.add_ranger_type(0)

    builder.add_tank_type(1)
    builder.add_sniper_type(1)
    builder.add_ranger_type(1)

    return GigastepEnv(**builder.get_kwargs())

def get_1v5_env():
    builder = ScenarioBuilder()
    builder.add_special_type(0)

    builder.add_default_type(1)
    builder.add_default_type(1)
    builder.add_default_type(1)
    builder.add_default_type(1)
    builder.add_default_type(1)

    return GigastepEnv(**builder.get_kwargs())