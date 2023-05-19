import jax.numpy as jnp
import numpy as onp

# TODO:
# Starting pos for each team: box or list of boxes for each agent?
# Starting heading for each team: float or range?
# Starting z for each team: float or range?

_builtin_maps = {
    "empty": {
        "boxes": jnp.zeros((0, 4), dtype=jnp.float32),
        "start_pos_team_a": jnp.array([0, 0, 3, 10], dtype=jnp.float32),
        "start_pos_team_b": jnp.array([7, 0, 10, 10], dtype=jnp.float32),
        "start_height": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_a": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_b": jnp.array([jnp.pi, jnp.pi], dtype=jnp.float32),
    },
    "two_rooms1": {
        "boxes": jnp.array(
            [
                [0, 4.8, 3, 5.2],
                [7, 4.8, 10, 5.2],
            ],
            dtype=jnp.float32,
        ),
        "start_pos_team_a": jnp.array([0, 0, 10, 3], dtype=jnp.float32),
        "start_pos_team_b": jnp.array([0, 7, 10, 10], dtype=jnp.float32),
        "start_height": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_a": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_b": jnp.array([jnp.pi, jnp.pi], dtype=jnp.float32),
    },
    "four_rooms": {
        "boxes": jnp.array(
            [
                [0, 4.8, 3, 5.2],
                [7, 4.8, 10, 5.2],
                [4.8, 0, 5.2, 3],
                [4.8, 7, 5.2, 10],
            ],
            dtype=jnp.float32,
        ),
        "start_pos_team_a": jnp.array([0, 0, 10, 3], dtype=jnp.float32),
        "start_pos_team_b": jnp.array([0, 7, 10, 10], dtype=jnp.float32),
        "start_height": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_a": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_b": jnp.array([jnp.pi, jnp.pi], dtype=jnp.float32),
    },
    "center_block": {
        "boxes": jnp.array(
            [
                [3.8, 3.8, 6.2, 6.2],
            ],
            dtype=jnp.float32,
        ),
        "start_pos_team_a": jnp.array([0, 0, 10, 3], dtype=jnp.float32),
        "start_pos_team_b": jnp.array([0, 7, 10, 10], dtype=jnp.float32),
        "start_height": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_a": jnp.array([0, 0], dtype=jnp.float32),
        "start_heading_team_b": jnp.array([jnp.pi, jnp.pi], dtype=jnp.float32),
    },
    
    # TODO: Upgrade the code below to also store the starting positions of the agents.
    "two_rooms2": jnp.array(
        [
            [4.8, 0, 5.2, 3],
            [4.8, 7, 5.2, 10],
        ],
        dtype=jnp.float32,
    ),
    "four_blocks": jnp.array(
        [
            [2.5, 2.5, 3.5, 3.5],
            [2.5, 6.5, 3.5, 7.5],
            [6.5, 2.5, 7.5, 3.5],
            [6.5, 6.5, 7.5, 7.5],
        ],
        dtype=jnp.float32,
    ),
    "cross": jnp.array(
        [
            [3.5, 4.8, 6.5, 5.2],
            [4.8, 3.5, 5.2, 6.5],
        ],
        dtype=jnp.float32,
    ),
    "s1": jnp.array(
        [
            [0, 3.1, 6.5, 3.5],
            [3.5, 6.4, 10, 6.8],
        ],
        dtype=jnp.float32,
    ),
    "s2": jnp.array(
        [
            [3.1, 0, 3.5, 6.5],
            [6.4, 3.5, 6.8, 10],
        ],
        dtype=jnp.float32,
    ),
    "tiles": jnp.array(
        [
            [0, 4.8, 2, 5.2],
            [8, 4.8, 10, 5.2],
            [4.8, 0, 5.2, 2],
            [4.8, 8, 5.2, 10],
        ],
        dtype=jnp.float32,
    ),
    "center_block": jnp.array(
        [
            [3.8, 3.8, 6.2, 6.2],
        ],
        dtype=jnp.float32,
    ),
    "center_block2": jnp.array(
        [
            [3, 3, 5, 5],
            [5, 5, 7, 7],
        ],
        dtype=jnp.float32,
    ),
    "center_block3": jnp.array(
        [
            [3, 5, 5, 7],
            [5, 3, 7, 5],
        ],
        dtype=jnp.float32,
    )
}


def _onp_draw_boxes(obs, boxes, resolution, limits):
    if boxes.shape[0] == 0:
        return obs
    x1, y1, x2, y2 = onp.split(boxes, 4, axis=-1)
    x1 = x1 * resolution[0] / limits[0]
    x2 = x2 * resolution[0] / limits[0]
    y1 = y1 * resolution[1] / limits[1]
    y2 = y2 * resolution[1] / limits[1]
    x1 = onp.clip(onp.int32(x1), 0, resolution[0] - 1)
    x2 = onp.clip(onp.int32(x2), 0, resolution[0] - 1)
    y1 = onp.clip(onp.int32(y1), 0, resolution[1] - 1)
    y2 = onp.clip(onp.int32(y2), 0, resolution[1] - 1)
    boxes = onp.concatenate([x1, y1, x2, y2], axis=-1)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        obs[x1:x2, y1:y2] = 127
    return obs


def get_builtin_maps(maps, limits):
    # Make sure all maps have the same size (for vmap)

    if maps == "all":
        list_of_maps = _builtin_maps
    elif maps == "empty":
        list_of_maps = {"empty": _builtin_maps["empty"]}
    else:
        raise ValueError(f"Unknown map name {maps}")

    max_map_size = max([m["boxes"].shape[0] for m in list_of_maps.values()])
    maps_boxes = []
    maps_start_pos_a = []
    maps_start_pos_b = []
    maps_start_height = []
    maps_heading_a = []
    maps_heading_b = []
    for k, v in list_of_maps.items():
        maps_boxes.append(
            jnp.pad(v["boxes"], ((0, max_map_size - v["boxes"].shape[0]), (0, 0)))
        )
        maps_start_pos_a.append(v["start_pos_team_a"])
        maps_start_pos_b.append(v["start_pos_team_b"])
        maps_heading_a.append(v["start_heading_team_a"])
        maps_heading_b.append(v["start_heading_team_b"])
        maps_start_height.append(v["start_height"])

    maps_boxes = jnp.stack(maps_boxes, axis=0)
    maps_start_pos_a = jnp.stack(maps_start_pos_a, axis=0)
    maps_start_pos_b = jnp.stack(maps_start_pos_b, axis=0)
    maps_heading_a = jnp.stack(maps_heading_a, axis=0)
    maps_heading_b = jnp.stack(maps_heading_b, axis=0)
    maps_start_height = jnp.stack(maps_start_height, axis=0)

    # maps are defined in [0,10]x[0,10] but we want to render them in [0,limits[0]]x[0,limits[1]]
    normalizer = jnp.array([[[10, 10, 10, 10]]])
    de_normalizer = jnp.array([[limits[0], limits[1], limits[0], limits[1]]])
    maps_boxes = maps_boxes / normalizer
    maps_boxes = maps_boxes * de_normalizer

    # One less dimension
    maps_start_pos_a = maps_start_pos_a / normalizer[0]
    maps_start_pos_a = maps_start_pos_a * de_normalizer[0]
    maps_start_pos_b = maps_start_pos_b / normalizer[0]
    maps_start_pos_b = maps_start_pos_b * de_normalizer[0]

    return {
        "boxes": maps_boxes,
        "start_pos_team_a": maps_start_pos_a,
        "start_pos_team_b": maps_start_pos_b,
        "start_height": maps_start_height,
        "start_heading_a": maps_heading_a,
        "start_heading_b": maps_heading_b,
    }


def prerender_maps(maps, resolution, limits):
    rendered_maps = []
    for map in maps:
        obs = onp.zeros([resolution[0], resolution[1], 3], dtype=jnp.uint8)

        # Draw map
        obs = _onp_draw_boxes(obs, map, resolution, limits)
        rendered_maps.append(obs)
    return jnp.array(onp.stack(rendered_maps))