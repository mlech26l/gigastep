import jax

import time
import jax.numpy as jnp

from gigastep import GigastepEnv, stack_agents, GigastepViewer, make_scenario
import cv2
from PIL import Image
import flax.linen as nn
import orbax.checkpoint


class MLP(nn.Module):  # create a Flax Module dataclass
    out_dims: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)  # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(256)(x)  # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.out_dims)(x)  # shape inference
        return x


# If you want to restore any of the leaves as sharded arrays, you'll need some
# extra arguments. See `PyTreeCheckpointHandler` documentation.
# restored = checkpointer.restore('path/to/directory/checkpoint_name/')


def loop_user():
    viewer = GigastepViewer(84 * 4, show_num_agents=0)
    viewer.set_title("User input")
    env = make_scenario("identical_5_vs_5_det", obs_type="vector")

    model = MLP(out_dims=3**3)  # instantiate the MLP model

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    params = {"params": checkpointer.restore("bc_policy.ckpt")}

    # env = make_scenario("waypoint_5_vs_5")
    rng = jax.random.PRNGKey(1)
    frames = []
    while True:
        rng, key = jax.random.split(rng, 2)
        state, obs = env.reset(key)
        t = 0
        ep_done = False
        while not ep_done:
            rng, key = jax.random.split(rng, 2)
            a1 = viewer.continuous_action
            a2 = model.apply(params, obs)
            a2 = jnp.argmax(a2, axis=-1)
            a2 = env.action_lut[a2]

            is_ego = jnp.arange(env.n_agents) == 0

            action = jnp.where(is_ego[:, None], a1, a2)
            rng, key = jax.random.split(rng, 2)
            state, obs, r, a, ep_done = env.step(state, action, key)
            print(f"Step {t:04d} reward {r[0]:0.1f}")
            frame_buffer = viewer.draw(env, state, None)
            t += 1
            # print(f"Waypoint: {state[1]['waypoint_location']}")
            # print(f"Ego position: {state[0]['x'][0]:0.2f}, {state[0]['y'][0]:0.2f}")
            # is_in_x = (
            #     state[0]["x"][0] >= state[1]["waypoint_location"][0]
            #     and state[0]["x"][0] <= state[1]["waypoint_location"][2]
            # )
            # is_in_y = (
            #     state[0]["y"][0] >= state[1]["waypoint_location"][1]
            #     and state[0]["y"][0] <= state[1]["waypoint_location"][3]
            # )
            # is_in_waypoint = is_in_x and is_in_y
            # if is_in_waypoint:
            #     print(f"Is in waypoint: {is_in_waypoint}")
            #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx")
            # else:
            #     print("X", is_in_x, "Y", is_in_y)
            # frames.append(Image.fromarray(frame_buffer))
            # print("len(frames)", len(frames))
            # if len(frames) == 200:
            #     print("saving gif")
            #     frame_one = frames[0]
            #     frame_one.save(
            #         "circle.gif",
            #         format="GIF",
            #         append_images=frames,
            #         save_all=True,
            #         duration=100,
            #         loop=0,
            #     )
            if viewer.should_quit:
                return
            if viewer.should_reset:
                print("resetting")
                break
            time.sleep(0.1)


if __name__ == "__main__":
    loop_user()