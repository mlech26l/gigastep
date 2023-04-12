import os
from functools import partial

import jax.numpy as jnp
import jax
import cv2

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from gigastep.builtin_maps import get_builtin_maps, prerender_maps
from gigastep.jax_utils import Discrete, Box


# from flax import struct



def stack_agents(*args):
    agents = jax.tree_map(lambda *xs: jnp.stack(xs, axis=0), *args)
    # map_state = {"boxes": jnp.zeros((0, 4), dtype=jnp.float32)}
    map_state = {"map_idx": jnp.int32(0)}
    # map_state = {"boxes": jnp.array([[4, 8.5, 6, 9]])}
    return (agents, map_state)


def draw_all_agents(obs, x, y, z, teams, alive, sprite):
    team1 = (teams == 1) * 255 * alive
    team2 = (teams == 0) * 255 * alive
    r = team1
    b = team2
    g = (z * alive).astype(jnp.uint8)  # // 2
    color = jnp.stack([r, g, b], axis=-1)
    sprite1_color = color * ((sprite & 1)[:, None] > 0)
    sprite2_color = color * ((sprite & 2)[:, None] > 0)
    sprite3_color = color * ((sprite & 4)[:, None] > 0)
    # Draw agents with different sprite
    obs = obs.at[x, y].max(sprite1_color, mode="drop")
    obs = obs.at[x + 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x - 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x, y + 1].max(sprite2_color, mode="drop")
    obs = obs.at[x, y - 1].max(sprite2_color, mode="drop")
    obs = obs.at[x + 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x + 1, y - 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y - 1].max(sprite3_color, mode="drop")

    return obs


def draw_agents_from_ego(obs, x, y, z, teams, seen, agent_id, sprite):
    team1 = (teams != teams[agent_id]) * 255 * seen
    team2 = (teams == teams[agent_id]) * 255 * seen
    seen_or_same_team = (seen > 0) | (teams == teams[agent_id])

    is_ego = jnp.arange(x.shape[0]) == agent_id
    r = team1 + is_ego * (255 - z)
    b = team2 * (1 - is_ego) + is_ego * (255 - z)
    g = z * seen * seen_or_same_team * (1 - is_ego) + 255 * is_ego
    color = jnp.stack([r, g, b], axis=-1)
    sprite1_color = color * ((sprite & 1)[:, None] > 0)
    sprite2_color = color * ((sprite & 2)[:, None] > 0)
    sprite3_color = color * ((sprite & 4)[:, None] > 0)

    # reset color at center of ego
    obs = obs.at[x[agent_id], y[agent_id], :].set(jnp.zeros(3, dtype=jnp.uint8))

    # Draw agents with different sprite
    obs = obs.at[x, y].max(sprite1_color, mode="drop")
    obs = obs.at[x + 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x - 1, y].max(sprite2_color, mode="drop")
    obs = obs.at[x, y + 1].max(sprite2_color, mode="drop")
    obs = obs.at[x, y - 1].max(sprite2_color, mode="drop")
    obs = obs.at[x + 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x + 1, y - 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y + 1].max(sprite3_color, mode="drop")
    obs = obs.at[x - 1, y - 1].max(sprite3_color, mode="drop")

    return obs


class GigastepEnv:
    def __init__(
        self,
        very_close_cone_depth=0.7,
        cone_depth=3.5,
        cone_angle=jnp.pi / 4,
        damage_per_second=3,
        healing_per_second=0.3,
        use_stochastic_obs=True,
        use_stochastic_comm=True,
        collision_range=0.4,
        collision_altitude=0.5,
        collision_penalty=10,
        limit_x = 10,
        limit_y = 10,
        resolution_x=84,
        resolution_y=84,
        n_agents=10,
        per_agent_sprites = None,
        per_agent_thrust = None,
        per_agent_max_health = None,
        per_agent_range = None,
        per_agent_team = None,
        tagged_penalty=5,
        discrete_actions=False,
        jit=True,
    ):
        self.n_agents = n_agents
        self.very_close_cone_depth = jnp.square(very_close_cone_depth)
        self.cone_depth = jnp.square(cone_depth)
        self.cone_angle = cone_angle
        self.damage_per_second = damage_per_second
        self.healing_per_second = healing_per_second
        self.collision_range = jnp.square(collision_range)
        self.collision_altitude = collision_altitude
        self.collision_penalty = collision_penalty
        self.tagged_penalty = tagged_penalty
        self.use_stochastic_obs = use_stochastic_obs
        self.use_stochastic_comm = use_stochastic_comm
        self.max_communication_range = 10

        if per_agent_sprites is None:
            per_agent_sprites = jnp.ones(n_agents, dtype=jnp.int32)
        self._per_agent_sprites = per_agent_sprites
        if per_agent_thrust is None:
            per_agent_thrust = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_thrust = per_agent_thrust
        if per_agent_max_health is None:
            per_agent_max_health = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_max_health = per_agent_max_health
        if per_agent_range is None:
            per_agent_range = jnp.ones(n_agents, dtype=jnp.float32)
        self._per_agent_range = per_agent_range

        if per_agent_team is None:
            team_blue = self.n_agents // 2
            team_red = self.n_agents - team_blue
            per_agent_team = jnp.concatenate(
                [jnp.ones((team_blue,)), jnp.zeros((team_red,))], axis=0
            )
        self._per_agent_team = per_agent_team


        self.limits = (limit_x, limit_y)
        self.z_min = 1
        self.z_max = 10
        self.resolution = (resolution_x, resolution_y)
        self.time_delta = 0.1
        self.observation_space = Box(
            low=jnp.zeros([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8),
            high=255
            * jnp.ones([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8),
        )
        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.action_space = Discrete(3**3)
            self.action_lut = jnp.array(
                jnp.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
            ).T.reshape(-1, 3)
        else:
            self.action_space = Box(low=-jnp.ones(3), high=jnp.ones(3))

        self._maps = get_builtin_maps(self.limits)
        self._prerendered_maps = prerender_maps(
            self._maps, self.resolution, self.limits
        )

        self._v_step_agents = jax.vmap(self._step_agents)
        self.v_step = jax.vmap(self.step)
        self.v_reset = jax.vmap(self.reset)
        if jit:
            self.v_reset = jax.jit(self.v_reset)
            self.v_step = jax.jit(self.v_step)

    @partial(jax.jit, static_argnums=(0,))
    def _step_agents(self, state, action):
        c_heading = 4
        c_dive = 5
        c_dive_throttle = 0.5
        c_throttle = 1.5
        v_min = 1.3
        g = 9.81
        v_resistance = 0.4

        if self.discrete_actions:
            action = self.action_lut[action]

        action = jnp.clip(action, -1, 1)
        u_heading, u_dive, u_throttle = action
        # x, y, z, v, heading, health, seen, alive, team = state
        alive = state["alive"]

        # Change heading and pitch based on action
        heading = state["heading"] + self.time_delta * u_heading * c_heading * alive
        # keep within pi
        heading = jnp.fmod(heading, 2 * jnp.pi)

        # Apply throttle
        v = (
            state["v"]
            + self.time_delta * u_throttle * c_throttle * alive * state["max_thrust"]
        )
        v = v - self.time_delta * v_resistance * jnp.square(v) * alive

        vx = v * jnp.cos(heading)
        vy = v * jnp.sin(heading)
        vz = c_dive * u_dive

        x = state["x"] + self.time_delta * vx * alive
        y = state["y"] + self.time_delta * vy * alive
        z = state["z"] + self.time_delta * vz * alive
        z = jnp.clip(z, self.z_min, self.z_max)

        z_delta = state["z"] - z
        v_new = jnp.sqrt(
            jnp.maximum(jnp.square(v) + c_dive_throttle * g * z_delta, jnp.square(v_min))
        )  # 0 at minimum
        # next_state = jnp.array([x, y, z, v_new, heading, health, seen, alive, team])
        next_state = {
            "x": x,
            "y": y,
            "z": z,
            "v": v_new,
            "heading": heading,
            "health": state["health"],
            # "seen": state["seen"],
            "alive": alive,
            "team": state["team"],
            "detection_range": state["detection_range"],
            "max_health": state["max_health"],
            "max_thrust": state["max_thrust"],
            "sprite": state["sprite"],
        }
        return next_state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, states, actions, rng):
        v_step = jax.vmap(self._step_agents)

        agent_states, map_state = states
        agent_states = v_step(agent_states, actions)

        # Check if agents are out of bounds
        # x, y, z, v, heading, health, seen, alive, teams = next_states.T
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = agent_states["team"]

        out_of_bounds = (x < 0) | (x > self.limits[0]) | (y < 0) | (y > self.limits[1])
        alive = alive * (1 - out_of_bounds)

        # Check if agents collided (cylinder shaped collision)
        collided = (
            (
                jnp.square(x[:, None] - x[None, :])
                + jnp.square(y[:, None] - y[None, :])
                < self.collision_range
            )
            & (jnp.abs(z[:, None] - z[None, :]) < self.collision_altitude)
            & (alive[:, None] == 1)
            & (alive[None, :] == 1)
        )
        collided = collided.astype(jnp.float32)
        collided = collided - jnp.diag(jnp.diag(collided))
        collided = jnp.sum(collided, axis=1) > 0
        alive = alive * (1 - collided)

        # A box is an array of [x1, y1, x2, y2]
        boxes = self._maps[map_state["map_idx"]]
        hit_box = (
            (x[:, None] > boxes[None, :, 0])
            & (x[:, None] < boxes[None, :, 2])
            & (y[:, None] > boxes[None, :, 1])
            & (y[:, None] < boxes[None, :, 3])
        )
        hit_box = jnp.sum(hit_box.astype(jnp.float32), axis=1) > 0
        hit_box = hit_box.astype(jnp.float32)
        alive = alive * (1 - hit_box)

        # Very close agents are surely detected
        very_close = (
            jnp.square(x[:, None] - x[None, :]) + jnp.square(y[:, None] - y[None, :])
            < self.very_close_cone_depth
        )
        # Agents in cone are detected with probability inversely proportional to distance
        closness_score = (
            jnp.square(x[:, None] - x[None, :]) + jnp.square(y[:, None] - y[None, :])
        ) / (self.cone_depth * agent_states["detection_range"][:, None])
        angles = jnp.arctan2(y[:, None] - y[None, :], x[:, None] - x[None, :])
        angles = (
            jnp.fmod(angles - agent_states["heading"][:, None] + jnp.pi, 2 * jnp.pi)
            - jnp.pi
        )
        in_cone = jnp.abs(angles) < self.cone_angle

        if self.use_stochastic_obs:
            # if not in cone, set closness_score to 1.1 so that it is always greater than rand
            in_cone_score = jnp.clip(closness_score, 0, 1) + (1 - in_cone) * 1.1
            rng, key = jax.random.split(rng)
            rand = jax.random.uniform(key, shape=in_cone_score.shape)
            stochastic_detected = in_cone_score < rand
        else:
            stochastic_detected = closness_score <= 1 & in_cone

        # Check if agents can see each other (not same team and alive)
        can_detect = (
            (teams[:, None] != teams[None, :])
            & (alive[:, None] == 1)
            & (alive[None, :] == 1)
        )
        # Probabilistic detection
        has_detected = can_detect & ((very_close == 1) | stochastic_detected)

        # Check if agents are in the cone of vision of another agent
        has_detected = has_detected.astype(jnp.float32)

        seen = jnp.sum(has_detected, axis=1)  # can be greater than 1
        health = (
            agent_states["health"]
            - seen * self.damage_per_second * self.time_delta
            + self.healing_per_second * self.time_delta
        )
        health = jnp.clip(
            health, 0, agent_states["max_health"]
        )  # max health is agent dependent

        detected_other_agent = jnp.sum(has_detected, axis=0)

        # Reward is proportional to the number of agents seen minus number of seen itself
        reward = (detected_other_agent - seen) * self.time_delta
        # Penalize for collisions or going out of bounds
        reward = reward - (collided + out_of_bounds + hit_box) * self.collision_penalty
        # Penalize for being tagged
        reward = reward - (health <= 0) * alive * self.tagged_penalty

        # Check if episode is done (all agents of one team dead)
        alive_team1 = jnp.sum(alive * (teams == 0))
        alive_team2 = jnp.sum(alive * (teams == 1))
        episode_done = (alive_team1 == 0) | (alive_team2 == 0)

        # Check if agents are dead
        alive = alive * (health > 0)

        # agent_states = jnp.stack(
        #     [x, y, z, v, heading, health, seen, alive, teams], axis=1
        # )
        agent_states = {
            "x": x,
            "y": y,
            "z": z,
            "v": agent_states["v"],
            "heading": agent_states["heading"],
            "health": health,
            # "seen": seen,
            "alive": alive,
            "team": teams,
            "max_thrust": agent_states["max_thrust"],
            "max_health": agent_states["max_health"],
            "detection_range": agent_states["detection_range"],
            "sprite": agent_states["sprite"],
        }
        next_states = (agent_states, map_state)
        v_get_observation = jax.vmap(
            self.get_observation, in_axes=(None, None, None, 0, 0)
        )
        rng = jax.random.split(rng, x.shape[0])
        obs = v_get_observation(
            next_states, has_detected, seen, rng, jnp.arange(x.shape[0])
        )
        alive = alive.astype(jnp.bool_)
        return next_states, obs, reward, alive, episode_done

    def get_dones(self, states):
        return states[0]["alive"]

    @partial(jax.jit, static_argnums=(0,))
    def get_observation(self, states, has_detected, took_damage, rng, agent_id):
        # x, y, z, v, heading, health, seen, alive, teams = states.T
        agent_states, map_state = states
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = agent_states["team"]
        # seen = agent_states["seen"]
        heading = agent_states["heading"]

        if self.use_stochastic_comm:
            distance = jnp.sqrt(
                jnp.square(x[agent_id, None] - x[None, :])
                + jnp.square(y[agent_id, None] - y[None, :])
            )
            distance = distance / self.max_communication_range
            distance = jnp.clip(distance, 0, 1)
            # Dead agents are out of communication range
            # print("distance1.shape", distance.shape)
            distance = (
                distance
                + (1 - alive[None, :]) * 2
                + (teams[agent_id] != teams[None, :])
            )
            # print("distance2.shape", distance.shape)
            rand = jax.random.uniform(rng, shape=distance.shape)
            communicate = distance < rand
            # print("communicate.shape", communicate.shape)

            seen = (has_detected + jnp.eye(has_detected.shape[0])) * communicate
        else:
            seen = has_detected + jnp.eye(has_detected.shape[0]) * (
                teams[agent_id] == teams[None, :]
            )
        # print("seen.shape", seen.shape)
        seen = jnp.sum(seen, axis=1) > 0
        # BUG: in debug scenario, ego is not alive -> seen is always 0
        # breakpoint()

        x = jnp.round(x * self.resolution[0] / self.limits[0]).astype(jnp.int32)
        y = jnp.round(y * self.resolution[1] / self.limits[1]).astype(jnp.int32)
        z = jnp.round((z - self.z_min) * 255 / (self.z_max - self.z_min)).astype(
            jnp.uint8
        )
        alive = alive.astype(jnp.uint8)
        teams = teams.astype(jnp.uint8)
        # obs = jnp.zeros([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8)
        obs = self._prerendered_maps[map_state["map_idx"]]

        # Border is red
        obs = obs.at[:, 0, 0].max(255)
        obs = obs.at[:, self.resolution[1] - 1, 0].max(255)
        obs = obs.at[0, :, 0].max(255)
        obs = obs.at[self.resolution[0] - 1, :, 0].max(255)
        # Border is white if agent is not taken damage
        not_taken_damage = (1 - took_damage[agent_id]).astype(jnp.uint8)
        obs = obs.at[:, 0, :].max(255 * not_taken_damage)
        obs = obs.at[:, self.resolution[1] - 1, :].max(255 * not_taken_damage)
        obs = obs.at[0, :, :].max(255 * not_taken_damage)
        obs = obs.at[self.resolution[0] - 1, :, :].max(255 * not_taken_damage)

        tail_length = 5
        own_team = (teams == teams[agent_id]).astype(jnp.uint8)

        # Draw tails first
        for i in range(1, tail_length):
            intensity = jnp.uint8(255 - 255 * i / tail_length)
            xi = x - i * jnp.cos(heading)
            yi = y - i * jnp.sin(heading)
            xi = jnp.clip(xi, 0, self.resolution[0] - 1).astype(jnp.int32)
            yi = jnp.clip(yi, 0, self.resolution[0] - 1).astype(jnp.int32)
            # Own team tail is blue
            obs = obs.at[xi, yi, 2].add(intensity * own_team * seen, mode="drop")
            # Other team tail is red
            obs = obs.at[xi, yi, 0].add(intensity * (1 - own_team) * seen, mode="drop")
            # Ego tail is white
            obs = obs.at[xi[agent_id], yi[agent_id], :].max(intensity)

        # Draw agents
        obs = draw_agents_from_ego(
            obs, x, y, z, teams, seen, agent_id, agent_states["sprite"]
        )
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def get_global_observation(self, states):
        agent_states, map_state = states
        x = agent_states["x"]
        y = agent_states["y"]
        z = agent_states["z"]
        alive = agent_states["alive"]
        teams = agent_states["team"]
        heading = agent_states["heading"]

        # x, y, z, v, heading, health, seen, alive, teams = states.T
        x = jnp.round(x * self.resolution[0] / self.limits[0]).astype(jnp.int32)
        y = jnp.round(y * self.resolution[1] / self.limits[1]).astype(jnp.int32)
        z = jnp.round((z - self.z_min) * 255 / (self.z_max - self.z_min)).astype(
            jnp.uint8
        )
        alive = alive.astype(jnp.uint8)
        teams = teams.astype(jnp.uint8)
        # obs = jnp.zeros([self.resolution[0], self.resolution[1], 3], dtype=jnp.uint8)
        obs = self._prerendered_maps[map_state["map_idx"]]

        # Draw border
        obs = obs.at[:, 0, :].max(255)
        obs = obs.at[:, self.resolution[1] - 1, :].max(255)
        obs = obs.at[0, :, :].max(255)
        obs = obs.at[self.resolution[0] - 1, :, :].max(255)

        # Draw boxes
        # obs = draw_boxes(obs, map_state["boxes"], self.resolution, self.limits)

        # Draw agents
        team1 = teams
        team2 = 1 - teams
        # Draw other agents

        tail_length = 5
        for i in range(1, tail_length):
            intensity = jnp.uint8(255 - 255 * i / tail_length)
            xi = x - i * jnp.cos(heading)
            yi = y - i * jnp.sin(heading)
            xi = jnp.clip(xi, 0, self.resolution[0] - 1).astype(jnp.int32)
            yi = jnp.clip(yi, 0, self.resolution[0] - 1).astype(jnp.int32)
            obs = obs.at[xi, yi, 0].add(intensity * team1 * alive, mode="drop")
            obs = obs.at[xi, yi, 2].add(intensity * team2 * alive, mode="drop")
        obs = draw_all_agents(obs, x, y, z, teams, alive, agent_states["sprite"])
        return obs

    def reset(self, rng):
        rng = jax.random.split(rng, 7)

        map_idx = jax.random.randint(
            rng[5], shape=(), minval=0, maxval=len(self._maps)
        )
        map_state = {
            "map_idx": map_idx,
        }

        x = jax.random.uniform(
            rng[0], shape=(self.n_agents,), minval=0, maxval=self.limits[0]
        )
        y = jax.random.uniform(
            rng[1], shape=(self.n_agents,), minval=0, maxval=self.limits[1]
        )
        z = jax.random.uniform(
            rng[2], shape=(self.n_agents,), minval=self.z_min, maxval=self.z_max
        )
        v = jax.random.uniform(rng[3], shape=(self.n_agents,), minval=1, maxval=2)
        heading = jax.random.uniform(
            rng[4], shape=(self.n_agents,), minval=0, maxval=2 * jnp.pi
        )
        for i in range(3):
            # slightly enlarge collision radius to avoid collisions in the first steps
            collision_enlarge = 0.1

            # Resample agents if there is a collision (with other agents or boxes)
            rng = jax.random.split(rng[-1], 4)
            collided = (
                jnp.square(x[:, None] - x[None, :])
                + jnp.square(y[:, None] - y[None, :])
                < self.collision_range + collision_enlarge
            ) & (
                jnp.abs(z[:, None] - z[None, :])
                < self.collision_altitude + collision_enlarge
            )
            collided = collided.astype(jnp.float32)
            collided = collided - jnp.diag(jnp.diag(collided))
            collided = jnp.sum(collided, axis=1) > 0

            boxes = self._maps[map_state["map_idx"]]
            hit_box = (
                (x[:, None] > boxes[None, :, 0] - collision_enlarge)
                & (x[:, None] < boxes[None, :, 2] + collision_enlarge)
                & (y[:, None] > boxes[None, :, 1] - collision_enlarge)
                & (y[:, None] < boxes[None, :, 3] + collision_enlarge)
            )
            hit_box = jnp.sum(hit_box.astype(jnp.float32), axis=1) > 0
            need_to_resample = collided | hit_box
            x_new = jax.random.uniform(
                rng[0], shape=(self.n_agents,), minval=0, maxval=self.limits[0]
            )
            y_new = jax.random.uniform(
                rng[1], shape=(self.n_agents,), minval=0, maxval=self.limits[1]
            )
            z_new = jax.random.uniform(
                rng[2], shape=(self.n_agents,), minval=self.z_min, maxval=self.z_max
            )
            x = jnp.where(need_to_resample, x_new, x)
            y = jnp.where(need_to_resample, y_new, y)
            z = jnp.where(need_to_resample, z_new, z)

        health = jnp.ones((self.n_agents,), dtype=jnp.float32)
        alive = jnp.ones((self.n_agents,), dtype=jnp.bool_)
        agent_state = {
            "x": x,
            "y": y,
            "z": z,
            "v": v,
            "heading": heading,
            "health": health,
            "alive": alive,
            "team": self._per_agent_team,
            "detection_range": self._per_agent_range,
            "max_health": self._per_agent_max_health,
            "max_thrust": self._per_agent_thrust,
            "sprite": self._per_agent_sprites,
        }

        state = (agent_state, map_state)

        v_get_observation = jax.vmap(
            self.get_observation, in_axes=(None, None, None, 0, 0)
        )
        rng = jax.random.split(rng[-1], x.shape[0])
        obs = v_get_observation(
            state, jnp.zeros((self.n_agents,self.n_agents)), jnp.zeros(self.n_agents), rng, jnp.arange(x.shape[0])
        )

        return state, obs

    @classmethod
    def get_initial_state(
        cls,
        x=0,
        y=0,
        z=0,
        v=1,
        heading=0,
        health=1,
        alive=1,
        team=0,
        max_health=1,
        detection_range=1,
        max_thrust=1,
        sprite=1,
    ):
        return {
            "x": x,
            "y": y,
            "z": z,
            "v": v,
            "heading": heading,
            "health": health,
            "alive": alive,
            "team": team,
            "max_health": max_health,
            "max_thrust": max_thrust,
            "detection_range": detection_range,
            "sprite": sprite,
        }

    @classmethod
    def action(cls, heading=0, dive=0, speed=0):
        return jnp.array([heading, dive, speed])