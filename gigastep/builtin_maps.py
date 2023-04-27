import jax.numpy as jnp
import numpy as onp

_builtin_maps = {
    "empty": jnp.zeros((0, 4), dtype=jnp.float32),
    "four_rooms": jnp.array(
        [
            [0, 4.8, 3, 5.2],
            [7, 4.8, 10, 5.2],
            [4.8, 0, 5.2, 3],
            [4.8, 7, 5.2, 10],
        ],
        dtype=jnp.float32,
    ),
    "two_rooms1": jnp.array(
        [
            [0, 4.8, 3, 5.2],
            [7, 4.8, 10, 5.2],
        ],
        dtype=jnp.float32,
    ),
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
    ),
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
        list_of_maps = {"empty": jnp.zeros((0, 4), dtype=jnp.float32)}
    else:
        raise ValueError(f"Unknown map name {maps}")

    max_map_size = max([m.shape[0] for m in list_of_maps.values()])
    maps = []
    for k, v in list_of_maps.items():
        maps.append(jnp.pad(v, ((0, max_map_size - v.shape[0]), (0, 0))))

    list_of_maps = jnp.stack(maps, axis=0)

    # maps are defined in [0,10]x[0,10] but we want to render them in [0,limits[0]]x[0,limits[1]]
    normalizer = jnp.array([[[10, 10, 10, 10]]])
    de_normalizer = jnp.array([[limits[0], limits[1], limits[0], limits[1]]])
    list_of_maps = list_of_maps / normalizer
    list_of_maps = list_of_maps * de_normalizer

    return list_of_maps


def prerender_maps(maps, resolution, limits):
    rendered_maps = []
    for map in maps:
        obs = onp.zeros([resolution[0], resolution[1], 3], dtype=jnp.uint8)

        # Draw map
        obs = _onp_draw_boxes(obs, map, resolution, limits)
        rendered_maps.append(obs)
    return jnp.array(onp.stack(rendered_maps))