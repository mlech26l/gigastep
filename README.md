# Gigastep - 1 billion steps per second multi-agent RL

![Gigastep](misc/scenario.webp)  

## Installation

```shell
pip3 install gigastep
```

## Features

- Collaborative and adversarial multi-agent  
- Partial observability (stochastic observations and communication)
- 3D dynamics
- Scalable (> 1000 agents, ```jax.jit``` and ```jax.vmap``` support)
- Heterogeneous agent types  

![Gigastep](misc/concat.webp)

## Usage

```python
from gigastep import GigastepEnv
import jax

n_agents = 20
dyn = GigastepEnv(n_agents=n_agents)
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)

ep_done = False
state = dyn.reset(key_reset)
while not ep_done:
    rng, key_action,key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(key_action, shape=(n_agents, 3), minval=-1, maxval=1)
    state, obs, rewards, alive, ep_done = dyn.step(state, action, key_step)
    # obs is an uint8 array of shape [n_agents, 84,84,3]
    # rewards is a float32 array of shape [n_agents]
    # alive is a bool array of shape [n_agents]
    # ep_done is a bool
```


## Vectorized Environment

The ```eng.reset``` and ```eng.step``` functions are vectorized using ```jax.vmap``` and 
accessible through the ```eng.v_reset``` and ```eng.v_step``` methods.

```python
from gigastep import GigastepEnv
import jax
import jax.numpy as jnp

n_agents = 20
batch_size = 32
dyn = GigastepEnv(n_agents=n_agents)
rng = jax.random.PRNGKey(3)
rng, key_reset = jax.random.split(rng, 2)
key_reset = jax.random.split(key_reset, batch_size)

state = dyn.v_reset(key_reset)
ep_dones = jnp.zeros(batch_size, dtype=jnp.bool_)
while not jnp.all(ep_dones):
    rng, key_action,key_step = jax.random.split(rng, 3)
    action = jax.random.uniform(key_action, shape=(batch_size, n_agents, 3), minval=-1, maxval=1)
    key_step = jax.random.split(key_step, batch_size)
    state, obs, rewards, alive, ep_dones = dyn.v_step(state, action, key_step)
    # obs is an uint8 array of shape [batch_size, n_agents, 84,84,3]
    # rewards is a float32 array of shape [batch_size, n_agents]
    # alive is a bool array of shape [batch_size, n_agents]
    # ep_done is a bool array of shape [batch_size]
```