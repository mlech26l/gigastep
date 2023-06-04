import glob
import os

import numpy as np
import optax
import jax.numpy as jnp
import jax
from flax.training.train_state import TrainState
import flax.linen as nn
from tqdm import tqdm

from gigastep import make_scenario
import tensorflow as tf
# disable gpu
tf.config.set_visible_devices([], 'GPU')

def read_files(path):
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    all_x, all_y = [], []
    for file in files:
        data = np.load(file)
        all_x.append(data["vec_obs"])
        all_y.append(data["action"])

    return np.concatenate(all_x,axis=0), np.concatenate(all_y,axis=0)

def create_train_state(module, rng, learning_rate, input_shape):
      """Creates an initial `TrainState`."""
      params = module.init(rng, jnp.ones(input_shape))['params']
      tx = optax.adam(learning_rate)
      return TrainState.create(
          apply_fn=module.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
      """Train for a single step."""
      def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch_x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch_y).mean()
            return loss, logits
      grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
      (_,logits), grads = grad_fn(state.params)
      state = state.apply_gradients(grads=grads)
      acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_y)
      return state, acc

ENV_KWARGS = {
"obs_type": "vector",  # "rgb", # "vector",
"discrete_actions": True,
"reward_game_won": 100,
"reward_defeat_one_opponent": 100,
"reward_detection": 0,
"reward_damage": 0,
"reward_idle": 0,
"reward_collision_agent": 0,
"reward_collision_obstacle": 100,
"cone_depth": 15.0,
"cone_angle": jnp.pi * 1.99,
"enable_waypoints": False,
"use_stochastic_obs": False,
"use_stochastic_comm": False,
"max_agent_in_vec_obs": 100,
"max_episode_length": 256,  # 1024,
}
def main():

    train_x, train_y = read_files("human_data")
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(10000).batch(32)
    env = make_scenario("identical_5_vs_5", **ENV_KWARGS)
    class MLP(nn.Module):  # create a Flax Module dataclass
        out_dims: int

        @nn.compact
        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            x = nn.Dense(128)(x)  # create inline Flax Module submodules
            x = nn.relu(x)
            x = nn.Dense(self.out_dims)(x)  # shape inference
            return x

    model = MLP(out_dims=env.action_space.n)  # instantiate the MLP model
    train_state = create_train_state(model, jax.random.PRNGKey(0), 3e-4, (1, env.observation_space.shape[0]))

    for epoch in range(100):
        total_acc = 0
        pbar = tqdm(tf.data.experimental.cardinality(train_ds).numpy())
        for batch_x, batch_y in train_ds.as_numpy_iterator():
            batch_x = jnp.array(batch_x)
            batch_y = jnp.array(batch_y)
            train_state, acc = train_step(train_state, batch_x, batch_y)
            total_acc += acc
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch}, accuracy: {total_acc*100 / (pbar.n + 1):0.2f}%")
        pbar.close()

if __name__ == "__main__":
    main()