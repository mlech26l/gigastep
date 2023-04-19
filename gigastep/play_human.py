import argparse

from gigastep import make_scenario, GigastepViewer
import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="identical_20_vs_20")
    args = parser.parse_args()

    viewer = GigastepViewer(frame_size=840, show_global_state=False, show_num_agents=1)
    env = make_scenario(args.scenario)
    rng = jax.random.PRNGKey(3)

    ep_idx = 0
    while True:
        rng, key_reset = jax.random.split(rng, 2)
        ep_done = False
        state, obs = env.reset(key_reset)
        total_user_reward = 0
        num_steps = 0
        while not ep_done:
            viewer.poll()
            if viewer.should_quit:
                return
            if viewer.should_reset:
                break

            key_action, key_step, key = jax.random.split(rng, 3)
            action_user = viewer.continuous_action
            action_ai = jax.random.uniform(
                key_action, shape=(env.n_agents - 1, 3), minval=-1, maxval=1
            )
            action = jnp.concatenate(
                [jnp.expand_dims(action_user, 0), action_ai], axis=0
            )
            state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
            img = viewer.draw(env, state, obs)
            viewer.clock.tick(10)
            total_user_reward += rewards[0]
            num_steps += 1
            if dones[0]:
                break
        ep_idx += 1
        print(
            f"Episode {ep_idx}: Total user reward: {total_user_reward:0.2f} in {num_steps} steps"
        )


if __name__ == "__main__":
    main()