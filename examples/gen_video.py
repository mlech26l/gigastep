import os
import time

import cv2
import jax
import numpy as np
import jax.numpy as jnp
import pygame

from gigastep import stack_agents, GigastepEnv, GigastepViewer

SLEEP_TIME = 0.01

# def start(self):
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#
#         self.display.blit(self.image, (0, 0))
#         pygame.display.update()
#
#


def loop_2agents():
    viewer = GigastepViewer(4 * 84, show_num_agents=2)
    rng = jax.random.PRNGKey(1)
    os.makedirs("video/2agents", exist_ok=True)
    s1 = GigastepEnv.get_initial_state(y=3, x=3, team=0)
    s2 = GigastepEnv.get_initial_state(y=6, x=7, heading=-jnp.pi / 2, team=1)
    # state = jnp.stack([s1, s2, s3], axis=0)
    state = stack_agents(s1, s2)
    dyn = GigastepEnv(n_agents=state[0]["x"].shape[0])
    frame = 0
    while jnp.sum(dyn.get_dones(state)) > 0:
        a1 = GigastepEnv.action(speed=1)
        a2 = GigastepEnv.action(speed=1)
        action = jnp.stack([a1, a2], axis=0)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        save_frame(dyn, state, obs, f"video/2agents/frame_{frame:04d}.png", obs2=True)
        frame += 1


def loop_some_agents():
    viewer = GigastepViewer(4 * 84, show_num_agents=3)
    rng = jax.random.PRNGKey(1)
    os.makedirs("video/some_agents", exist_ok=True)
    s1 = GigastepEnv.get_initial_state(y=5, x=1, team=0)
    s2 = GigastepEnv.get_initial_state(y=5, x=2, team=1)
    s3 = GigastepEnv.get_initial_state(y=4, x=2, team=1)
    s6 = GigastepEnv.get_initial_state(y=3, x=2, team=1)
    s4 = GigastepEnv.get_initial_state(y=4.5, x=2, team=1)
    s5 = GigastepEnv.get_initial_state(y=3.5, x=2, team=1)
    state = stack_agents(s1, s2, s3, s4, s5, s6)
    dyn = GigastepEnv(n_agents=state[0]["x"].shape[0])
    frame = 0
    while jnp.sum(dyn.get_dones(state)) > 0:
        action = jnp.zeros((6, 3))
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        save_frame(
            dyn, state, obs, f"video/some_agents/frame_{frame:04d}.png", obs2=True
        )
        frame += 1


def save_frame(
    dyn, state, step_obs, filename, frame_size=4 * 84, obs1=True, obs2=False
):
    obs = dyn.get_global_observation(state)
    obs = np.array(obs, dtype=np.uint8)
    # obs = cv2.cvtColor(np.array(obs), cv2.COLOR_RGB2BGR)
    obs = cv2.resize(obs, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)

    image = obs
    if obs1:
        obs_1 = step_obs[0]
        obs_1 = np.array(obs_1, dtype=np.uint8)
        # obs_1 = cv2.cvtColor(np.array(obs_1), cv2.COLOR_RGB2BGR)
        obs_1 = cv2.resize(
            obs_1, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_1], axis=1)
    if obs2:
        obs_2 = step_obs[1]
        obs_2 = np.array(obs_2, dtype=np.uint8)
        obs_2 = cv2.resize(
            obs_2, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST
        )
        image = np.concatenate([image, obs_2], axis=1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image)


def loop_random_agents():
    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    os.makedirs("video/random", exist_ok=True)
    viewer = GigastepViewer(4 * 84, show_num_agents=3)

    frame = 0
    t = 0
    while frame < 120:
        obs, state = dyn.reset(key)
        while jnp.sum(dyn.get_dones(state)) > 0:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            save_frame(
                dyn,
                state,
                obs,
                f"video/random/frame_{frame:04d}.png",
                obs1=True,
                obs2=True,
            )
            viewer.draw(dyn, state, obs)
            frame += 1
            t += 1
            if t > 20:
                t = 0
                break


def loop_random_many_agents():
    n_agents = 2000
    res = 200
    dyn = GigastepEnv(
        n_agents=n_agents, limit_x=50, limit_y=50, resolution_x=res, resolution_y=res
    )
    rng = jax.random.PRNGKey(3)

    key, rng = jax.random.split(rng, 2)
    os.makedirs("video/many", exist_ok=True)
    viewer = GigastepViewer(2 * res, show_num_agents=3)

    obs, state = dyn.reset(key)
    frame = 0
    while jnp.sum(dyn.get_dones(state)) > 0:
        rng, key = jax.random.split(rng, 2)
        action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        save_frame(dyn, state, obs, f"video/many/frame_{frame:04d}.png")
        viewer.draw(dyn, state, obs)
        frame += 1


def loop_visible_debug():
    viewer = GigastepViewer(4 * 84, show_num_agents=1)

    rng = jax.random.PRNGKey(3)
    os.makedirs("video/visibility", exist_ok=True)
    frame = 0
    while True:
        state = [GigastepEnv.get_initial_state(y=5, x=5, v=0, team=0)]
        for i in range(40):
            for j in range(19):
                state.append(
                    GigastepEnv.get_initial_state(
                        y=3 + 0.25 * j, x=3 + 0.25 * i, v=0, team=1, z=10
                    )
                )
        state = stack_agents(*state)
        dyn = GigastepEnv(
            collision_range=0.0,
            use_stochastic_obs=True,
            damage_per_second=0,
            n_agents=state[0]["x"].shape[0],
        )  # disable collision
        # state = jnp.stack(state, axis=0)
        action = [GigastepEnv.action(speed=0) for s in range(state[0]["x"].shape[0])]
        action = jnp.stack(action, axis=0)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        save_frame(dyn, state, obs, f"video/visibility/frame_{frame:04d}.png")
        frame += 1
        if frame > 100:
            break


def loop_2agents_altitude():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("3 agents, same thrust but different altitude")
    rng = jax.random.PRNGKey(1)
    frame = 0
    os.makedirs("video/dive", exist_ok=True)
    s1 = GigastepEnv.get_initial_state(y=3, team=0)
    s2 = GigastepEnv.get_initial_state(y=5, team=0, z=10)
    s3 = GigastepEnv.get_initial_state(y=7, team=0, z=0)
    # state = jnp.stack([s1, s2, s3], axis=0)
    state = stack_agents(s1, s2, s3)
    dyn = GigastepEnv(n_agents=state[0]["x"].shape[0])

    t = 0
    while jnp.sum(dyn.get_dones(state)) > 0:
        a1 = GigastepEnv.action(speed=1)
        a2 = GigastepEnv.action(speed=1, dive=-1 if t > 10 else 0)
        a3 = GigastepEnv.action(speed=1, dive=1 if t > 10 else 0)
        t = t + 1
        action = jnp.stack([a1, a2, a3], axis=0)
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        save_frame(dyn, state, obs, f"video/dive/frame_{frame:04d}.png", obs1=False)
        frame += 1
        if frame > 100:
            break


def loop_maps():
    viewer = GigastepViewer(84 * 4, show_num_agents=3)
    viewer.set_title("10 random agents")
    n_agents = 20
    dyn = GigastepEnv(n_agents=n_agents)
    rng = jax.random.PRNGKey(4)
    os.makedirs("video/maps", exist_ok=True)

    key, rng = jax.random.split(rng, 2)
    frame = 0
    while True:
        obs, state = dyn.reset(key)
        t = 0
        while jnp.sum(dyn.get_dones(state)) > 0:
            rng, key = jax.random.split(rng, 2)
            action = jax.random.uniform(key, shape=(n_agents, 3), minval=-1, maxval=1)
            rng, key = jax.random.split(rng, 2)
            obs, state, r, a, d = dyn.step(state, action, key)
            viewer.draw(dyn, state, obs)
            save_frame(dyn, state, obs, f"video/maps/frame_{frame:04d}.png", obs1=True)
            frame += 1
            if frame >= 150:
                return
            t += 1
            if t > 10:
                break


def loop_heterogenous():
    viewer = GigastepViewer(84 * 4, show_num_agents=3)
    viewer.set_title("3 agents, same thrust but different altitude")
    dyn = GigastepEnv(
        use_stochastic_obs=False, use_stochastic_comm=False, very_close_cone_depth=10
    )
    rng = jax.random.PRNGKey(1)
    frame = 0
    os.makedirs("video/hetero", exist_ok=True)

    for i in range(8):
        s1 = GigastepEnv.get_initial_state(
            x=5, y=5, team=0, heading=np.pi / 4, sprite=i
        )
        s2 = GigastepEnv.get_initial_state(
            x=7, y=3, team=0, heading=np.pi / 4, sprite=i + 3 % 8
        )
        s3 = GigastepEnv.get_initial_state(
            x=3, y=7, team=1, heading=np.pi / 4, sprite=i + 5 % 8
        )
        state = stack_agents(s1, s2, s3)
        action = jnp.zeros((3, 3))
        rng, key = jax.random.split(rng, 2)
        obs, state, r, a, d = dyn.step(state, action, key)
        viewer.draw(dyn, state, obs)
        for i in range(5):
            save_frame(
                dyn, state, obs, f"video/hetero/frame_{frame:04d}.png", obs1=True
            )
            frame += 1


if __name__ == "__main__":
    # loop_some_agents()
    # loop_2agents_altitude()
    # loop_2agents()
    # loop_visible_debug()
    loop_random_agents()
    loop_random_many_agents()
    loop_maps()
    loop_heterogenous()
    # convert -delay 10 -loop 0 video/random/frame_*.png video/random.gif
    # convert -delay 10 -loop 0 video/2agents/frame_*.png video/2agents.gif
    # convert -delay 10 -loop 0 video/visibility/frame_*.png video/visibility.gif
    # convert -delay 10 -loop 0 video/many/frame_*.png video/many.gif
    # convert -delay 10 -loop 0 video/maps/frame_*.png video/maps.gif
    # convert -delay 50 -loop 0 video/hetero/frame_*.png video/hetero.gif