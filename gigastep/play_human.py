import argparse
import time

from gigastep import make_scenario, GigastepViewer
import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="identical_5_vs_5")
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
            if viewer.should_pause:
                continue

            key_action, key_step, key = jax.random.split(rng, 3)
            action_user = viewer.continuous_action
            print("action_user", action_user)
            action_ai = jax.random.uniform(
                key_action, shape=(env.n_agents - 1, 3), minval=-1, maxval=1
            )
            action_ai = jnp.zeros((env.n_agents - 1, 3))
            action = jnp.concatenate(
                [jnp.expand_dims(action_user, 0), action_ai], axis=0
            )
            state, obs, rewards, dones, ep_done = env.step(state, action, key_step)
            img = viewer.draw(env, state, obs)
            viewer.clock.tick(10)
            total_user_reward += rewards[0]
            num_steps += 1
            if dones[0] or ep_done:
                break
        if dones[0]:
            print("User died!")
            # Rumble 1 time with left motor if the episode is lost
            viewer.joystick.vibrate(1, 0, 400)
        elif ep_done:
            # Rumble 3 times with right motor if the episode is won
            print("Episode done!")
            for i in range(3):
                viewer.joystick.vibrate(0, 1, 100)
                time.sleep(0.12)
                viewer.pygame.event.pump()
                time.sleep(0.12)
        ep_idx += 1
        print(
            f"Episode {ep_idx}: Total user reward: {total_user_reward:0.2f} in {num_steps} steps"
        )


if __name__ == "__main__":
    main()