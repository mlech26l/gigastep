import jax

import time
import jax.numpy as jnp

from gigastep import GigastepEnv, stack_agents, GigastepViewer
import cv2
from PIL import Image


def loop_user():
    viewer = GigastepViewer(84 * 4)
    viewer.set_title("User input")
    dyn = GigastepEnv(
        # collision_range=0.0,
        # min_tracking_time=3,
        # damage_per_second=1,
        # max_tracking_time=10,
        # damage_cone_depth=4.2,
    )
    rng = jax.random.PRNGKey(1)
    frames = []
    while True:
        s1 = GigastepEnv.get_initial_state(x=3, y=3, team=1)
        s2 = GigastepEnv.get_initial_state(x=7, y=1, team=0)
        s3 = GigastepEnv.get_initial_state(x=8, y=6, team=0)
        s4 = GigastepEnv.get_initial_state(x=6, y=4, team=0)
        s5 = GigastepEnv.get_initial_state(x=9, y=4.4, team=0)
        s6 = GigastepEnv.get_initial_state(x=6.5, y=8, team=0)
        s7 = GigastepEnv.get_initial_state(x=7.5, y=5.6, team=0)
        s8 = GigastepEnv.get_initial_state(x=8.5, y=2.1, team=0)
        s9 = GigastepEnv.get_initial_state(x=7.8, y=9.8, team=0)
        # state = jnp.stack([s1, s2, s3], axis=0)
        state = stack_agents(s1, s2, s3, s4, s5, s6, s7, s8, s9)
        t = 0
        ep_done = False
        while not ep_done:
            rng, key = jax.random.split(rng, 2)
            a1 = viewer.continuous_action
            a2 = jnp.repeat(jnp.array([[1, 0, 0]]), 8, axis=0)
            action = jnp.concatenate([a1[None, :], a2], axis=0)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, ep_done = dyn.step(state, action, key)
            frame_buffer = viewer.draw(dyn, state, obs)
            frames.append(Image.fromarray(frame_buffer))
            print("len(frames)", len(frames))
            if len(frames) == 200:
                print("saving gif")
                frame_one = frames[0]
                frame_one.save(
                    "circle.gif",
                    format="GIF",
                    append_images=frames,
                    save_all=True,
                    duration=100,
                    loop=0,
                )
            if viewer.should_quit:
                return
            time.sleep(0.1)


if __name__ == "__main__":
    loop_user()