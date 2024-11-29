"""Code for generating grid POMDPs"""

import stormvogel.model

WALL = 999
NOT_WALL = 0

LEFT = "←"
RIGHT = "→"
UP = "↑"
DOWN = "↓"


def direction_result(x: int, y: int, direction: str, width: int, height: int):
    d = {UP: (0, -1), RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0)}
    res_x = x + d[direction][0]
    res_y = y + d[direction][1]
    if res_x < 0:
        return ((0, res_y), WALL)
    if res_x >= width:
        return ((width - 1, res_y), WALL)
    if res_y < 0:
        return ((res_x, 0), WALL)
    if res_y >= height:
        return ((res_x, height - 1), WALL)

    return ((res_x, res_y), NOT_WALL)


def grid_world(width: int, height: int, position_scalar: int = 200):
    """Create a grid world with an actor."""
    pomdp = stormvogel.model.new_pomdp(create_initial_state=False)
    grid = {
        (x, y): pomdp.new_state(f"({x},{y})")
        for x in range(width)
        for y in range(height)
    }
    dirs = {d: pomdp.new_action(d) for d in [UP, DOWN, LEFT, RIGHT]}
    positions = {}
    # Add movement
    for x in range(width):
        for y in range(height):
            for d, action in dirs.items():
                ((res_x, res_y), observation) = direction_result(x, y, d, width, height)
                state = grid[(x, y)]

                took_dir = pomdp.new_state(f"({x},{y}) {d}")
                positions[str(state.id)] = {
                    "x": x * position_scalar,
                    "y": y * position_scalar,
                }

                state.add_transitions([(action, took_dir)])
                took_dir.add_transitions([(1, grid[(res_x, res_y)])])
                took_dir.set_observation(observation)
    return pomdp, positions
