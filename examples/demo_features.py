import sys
import time

import jax
import jax.numpy as jnp
from gigastep import GigastepEnv, stack_agents
from gigastep import GigastepViewer


SLEEP_TIME = 0.01


def loop_2agents_altitude():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("3 agents, same thrust but different altitude")
    dyn = GigastepEnv()
    rng = jax.random.PRNGKey(1)
    while True:
        s1 = GigastepEnv.get_initial_state(y=3, team=1)
        s2 = GigastepEnv.get_initial_state(y=5, team=1, z=dyn.z_max)
        s3 = GigastepEnv.get_initial_state(y=7, team=1, z=0)
        # state = jnp.stack([s1, s2, s3], axis=0)
        state = stack_agents(s1, s2, s3)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=1)
            a2 = GigastepEnv.action(speed=1, dive=-1 if t > 10 else 0)
            a3 = GigastepEnv.action(speed=1, dive=1 if t > 10 else 0)
            action = jnp.stack([a1, a2, a3], axis=0)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            time.sleep(SLEEP_TIME)


def loop_speed_up_slow_down():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("3 agents, different thrust")
    dyn = GigastepEnv()
    rng = jax.random.PRNGKey(1)
    while True:
        s1 = GigastepEnv.get_initial_state(y=3, team=1)
        s2 = GigastepEnv.get_initial_state(y=5, team=1)
        s3 = GigastepEnv.get_initial_state(y=7, team=1)
        state = stack_agents(s1, s2, s3)
        # state = jnp.stack([s1, s2, s3], axis=0)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=0)
            a2 = GigastepEnv.action(speed=-1 if t > 10 else 0)
            a3 = GigastepEnv.action(speed=1 if t > 10 else 0)
            action = jnp.stack([a1, a2, a3], axis=0)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            time.sleep(SLEEP_TIME)


def loop_collide_direct():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("Agents colliding or not depending on altitude")
    dyn = GigastepEnv()
    rng = jax.random.PRNGKey(1)
    while True:
        s1 = GigastepEnv.get_initial_state(y=2, team=1)
        s2 = GigastepEnv.get_initial_state(y=2, x=dyn.limits[0], heading=jnp.pi, team=0)
        s3 = GigastepEnv.get_initial_state(y=4, team=1)
        s4 = GigastepEnv.get_initial_state(
            y=4, x=dyn.limits[0], heading=jnp.pi, team=0, z=dyn.z_max
        )
        s5 = GigastepEnv.get_initial_state(y=6, team=1)
        s6 = GigastepEnv.get_initial_state(y=6, x=dyn.limits[0], heading=jnp.pi, team=1)
        s7 = GigastepEnv.get_initial_state(y=8, team=1)
        s8 = GigastepEnv.get_initial_state(
            y=8, x=dyn.limits[0], heading=jnp.pi, team=1, z=dyn.z_max
        )
        state = stack_agents(s1, s2, s3, s4, s5, s6, s7, s8)
        # state = jnp.stack([s1, s2, s3, s4, s5, s6, s7, s8], axis=0)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=1)
            a2 = GigastepEnv.action(speed=1)
            a3 = GigastepEnv.action(speed=1)
            a4 = GigastepEnv.action(speed=1)
            a5 = GigastepEnv.action(speed=1)
            a6 = GigastepEnv.action(speed=1)
            a7 = GigastepEnv.action(speed=1)
            a8 = GigastepEnv.action(speed=1)
            action = jnp.stack([a1, a2, a3, a4, a5, a6, a7, a8], axis=0)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            time.sleep(SLEEP_TIME)


def loop_random_agents():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("10 random agents")
    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    while True:
        obs, state = dyn.reset(key)
        while jnp.sum(dyn.get_dones(state)) > 0:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)


def loop_reset_states():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("10 random agents")
    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    rng = jax.random.PRNGKey(3)

    while True:
        key, rng = jax.random.split(rng, 2)
        obs, state = dyn.reset(key)
        viewer.draw(dyn, state, obs)
        if viewer.should_pause:
            return
        if viewer.should_quit:
            sys.exit(1)
        time.sleep(0.3)


def loop_visible_debug():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("Radar visibility cone")
    dyn = GigastepEnv(
        collision_range=0.0, use_stochastic_obs=True, damage_per_second=0
    )  # disable collision
    rng = jax.random.PRNGKey(3)
    t = 0
    while True:
        rot = jnp.fmod(t * 0.4 + jnp.pi / 2, 2 * jnp.pi)
        print(f"Heading {rot * 180 / jnp.pi:0.1f} degree")
        state = [GigastepEnv.get_initial_state(y=5, x=5, v=0, team=0, heading=rot)]
        for i in range(40):
            for j in range(19):
                state.append(
                    GigastepEnv.get_initial_state(
                        y=3 + 0.25 * j, x=3 + 0.25 * i, v=0, team=1, z=dyn.z_max
                    )
                )
        state = stack_agents(*state)
        # state = jnp.stack(state, axis=0)
        action = [GigastepEnv.action(speed=0) for s in range(state[0]["x"].shape[0])]
        action = jnp.stack(action, axis=0)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        if viewer.should_pause:
            return
        if viewer.should_quit:
            sys.exit(1)
        time.sleep(SLEEP_TIME)
        t += 1


def loop_agent_sprites():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("Radar visibility cone")
    dyn = GigastepEnv(collision_range=0.0, use_stochastic_obs=True)  # disable collision
    rng = jax.random.PRNGKey(3)
    z = dyn.z_min
    while True:
        z += 1
        if z > dyn.z_max:
            z = dyn.z_min
        state = [
            GigastepEnv.get_initial_state(
                y=6, x=1, z=z, heading=jnp.pi / 4, team=0, sprite=5
            ),
            GigastepEnv.get_initial_state(
                y=2, x=3, v=0, heading=jnp.pi / 4, team=1, sprite=1
            ),
            GigastepEnv.get_initial_state(
                y=2, x=7, v=0, heading=jnp.pi / 4, team=0, sprite=1
            ),
            GigastepEnv.get_initial_state(
                y=2, x=1, z=dyn.z_max, heading=jnp.pi / 4, team=1, sprite=1
            ),
            GigastepEnv.get_initial_state(
                y=2, x=9, z=dyn.z_max, heading=jnp.pi / 4, team=0, sprite=1
            ),
            GigastepEnv.get_initial_state(
                y=4, x=3, v=0, heading=jnp.pi / 4, team=0, sprite=3
            ),
            GigastepEnv.get_initial_state(
                y=4, x=7, v=0, heading=jnp.pi / 4, team=1, sprite=3
            ),
            GigastepEnv.get_initial_state(
                y=4, x=1, z=dyn.z_max, heading=jnp.pi / 4, team=0, sprite=3
            ),
            GigastepEnv.get_initial_state(
                y=4, x=9, z=dyn.z_max, heading=jnp.pi / 4, team=1, sprite=3
            ),
            GigastepEnv.get_initial_state(
                y=6, x=3, v=0, heading=jnp.pi / 4, team=0, sprite=5
            ),
            GigastepEnv.get_initial_state(
                y=6, x=7, v=0, heading=jnp.pi / 4, team=1, sprite=5
            ),
            GigastepEnv.get_initial_state(
                y=6, x=9, z=dyn.z_max, heading=jnp.pi / 4, team=1, sprite=5
            ),
            GigastepEnv.get_initial_state(
                y=8, x=3, v=0, heading=jnp.pi / 4, team=0, sprite=7
            ),
            GigastepEnv.get_initial_state(
                y=8, x=7, v=0, heading=jnp.pi / 4, team=1, sprite=7
            ),
            GigastepEnv.get_initial_state(
                y=8, x=1, z=dyn.z_max, heading=jnp.pi / 4, team=0, sprite=7
            ),
            GigastepEnv.get_initial_state(
                y=8, x=9, z=dyn.z_max, heading=jnp.pi / 4, team=1, sprite=7
            ),
        ]
        state = stack_agents(*state)
        # state = jnp.stack(state, axis=0)
        # rng, key = jax.random.split(rng, 2)
        # action = jax.random.uniform(
        #     key, shape=(state[0]["x"].shape[0], 3), minval=-1, maxval=1
        # )
        action = jnp.zeros((state[0]["x"].shape[0], 3))
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        if viewer.should_pause:
            return
        if viewer.should_quit:
            sys.exit(1)
        time.sleep(SLEEP_TIME)


def loop_collision_with_offset():
    viewer = GigastepViewer(84 * 4, show_num_agents=2)
    viewer.set_title("Agents colliding or not depending on distance")
    dyn = GigastepEnv(use_stochastic_obs=False)
    rng = jax.random.PRNGKey(3)
    while True:
        s1 = GigastepEnv.get_initial_state(y=2, team=1)
        s2 = GigastepEnv.get_initial_state(
            y=2.1, x=dyn.limits[0], heading=jnp.pi, team=0, detection_range=100
        )
        s3 = GigastepEnv.get_initial_state(y=4, team=1)
        s4 = GigastepEnv.get_initial_state(
            y=4.2, x=dyn.limits[0], heading=jnp.pi, team=0
        )
        s5 = GigastepEnv.get_initial_state(y=6, team=1)
        s6 = GigastepEnv.get_initial_state(
            y=6.3, x=dyn.limits[0], heading=jnp.pi, team=1
        )
        s7 = GigastepEnv.get_initial_state(y=8, team=1)
        s8 = GigastepEnv.get_initial_state(
            y=8.4, x=dyn.limits[0], heading=jnp.pi, team=1
        )
        state = stack_agents(s1, s2, s3, s4, s5, s6, s7, s8)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=1)
            a2 = GigastepEnv.action(speed=1)
            a3 = GigastepEnv.action(speed=1)
            a4 = GigastepEnv.action(speed=1)
            a5 = GigastepEnv.action(speed=1)
            a6 = GigastepEnv.action(speed=1)
            a7 = GigastepEnv.action(speed=1)
            a8 = GigastepEnv.action(speed=1)
            action = jnp.stack([a1, a2, a3, a4, a5, a6, a7, a8], axis=0)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            # time.sleep(SLEEP_TIME)
            time.sleep(0.1)


def loop_maps():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("10 random agents")
    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    while True:
        obs, state = dyn.reset(key)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            time.sleep(SLEEP_TIME)
            t += 1
            if t > 10:
                break


def loop_heading():
    viewer = GigastepViewer(84 * 4, show_num_agents=2)
    viewer.set_title("Agents colliding or not depending on distance")
    dyn = GigastepEnv(use_stochastic_obs=False)
    rng = jax.random.PRNGKey(3)
    while True:
        s1 = GigastepEnv.get_initial_state(y=2, team=1)
        s2 = GigastepEnv.get_initial_state(
            y=5.1, x=0, heading=jnp.pi / 4, team=0, detection_range=1
        )
        state = stack_agents(s1, s2)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            a1 = GigastepEnv.action(speed=1)
            a2 = GigastepEnv.action(speed=1)
            action = jnp.stack([a1, a2], axis=0)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            print("Health", state[0]["health"])
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            t += 1
            # time.sleep(SLEEP_TIME)
            time.sleep(0.1)


def loop_communication():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("10 random agents")
    n_agents = 20
    dyn = GigastepEnv(damage_per_second=0, use_stochastic_comm=True, n_agents=n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    while True:
        obs, state = dyn.reset(key)
        while jnp.sum(dyn.get_dones(state)) > 0:
            action = jnp.zeros((n_agents, 3))
            rng, key = jax.random.split(rng, 2)
            # Don't update state
            obs, _state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            if viewer.should_pause:
                return
            if viewer.should_quit:
                sys.exit(1)
            if viewer.should_reset:
                rng, key = jax.random.split(rng, 2)
                obs, state = dyn.reset(key)
            time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    loop_collision_with_offset()
    loop_collide_direct()
    loop_2agents_altitude()
    loop_visible_debug()
    loop_heading()
    loop_reset_states()
    loop_random_agents()
    loop_communication()
    loop_maps()
    loop_agent_sprites()
    loop_speed_up_slow_down()