import sys
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv, stack_agents
from gigastep import GigastepViewer


SLEEP_TIME = 0.01

def loop_throttle():
    viewer = GigastepViewer(84 * 4,show_num_agents=0)
    viewer.set_title("3 agents, same thrust but different altitude")
    dyn = GigastepEnv()
    rng = jax.random.PRNGKey(1)
    while True:
        s1 = GigastepEnv.get_initial_state(x=1, y=1, team=1)
        s2 = GigastepEnv.get_initial_state(x=1, y=2, team=1)
        s3 = GigastepEnv.get_initial_state(x=1, y=3, team=1)
        s4 = GigastepEnv.get_initial_state(x=1, y=4, team=1, z=dyn.z_max)
        s5 = GigastepEnv.get_initial_state(x=1, y=5, team=1, z=0)
        s6 = GigastepEnv.get_initial_state(x=1, y=6, team=1, z=dyn.z_max)
        s7 = GigastepEnv.get_initial_state(x=1, y=7, team=1, z=0)
        state = stack_agents(s1, s2, s3,s4,s5,s6,s7)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=0)
            a2 = GigastepEnv.action(speed=1)
            a3 = GigastepEnv.action(speed=-1)
            a4 = GigastepEnv.action(speed=1,dive=-1)
            a5 = GigastepEnv.action(speed=1,dive=1)
            a6 = GigastepEnv.action(speed=0, dive=-1)
            a7 = GigastepEnv.action(speed=-1, dive=1)
            action = jnp.stack([a1, a2, a3,a4,a5,a6,a7], axis=0)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_reset:
                break
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            time.sleep(SLEEP_TIME)




if __name__ == "__main__":
    loop_throttle()