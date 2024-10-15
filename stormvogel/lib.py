from typing import Any
import stormvogel.model
import matplotlib.pyplot as plt


def naive_value_iteration(
    model: stormvogel.model.Model, steps: int, starting_state: stormvogel.model.State
) -> list[list[float]]:
    """Run naive value iteration. The result is a 2D list where result[n][m] is the probability to be in state m at step n.

    Args:
        model (stormvogel.model.Model): Target model.
        steps (int): Amount of steps.
        starting_state (stormvogel.model.State): Starting state.

    Returns:
        list[list[float]]: The result is a 2D list where result[n][m] is the probability to be in state m at step n.
    """
    if steps < 2:
        RuntimeError("Need at least two steps")
    if model.type != stormvogel.model.ModelType.DTMC:
        RuntimeError("Only works for DTMC")

    # Create a matrix and set the value for the starting state to 1 on the first step.
    matrix_steps_states = [[0.0 for s in model.states] for x in range(steps)]
    matrix_steps_states[0][starting_state.id] = 1

    # Apply the updated values for each step.
    for current_step in range(steps - 1):
        next_step = current_step + 1
        for s_id, s in model.get_states().items():
            branch = model.get_branch(s)
            for transition_prob, target in branch.branch:
                current_prob = matrix_steps_states[current_step][s_id]
                matrix_steps_states[next_step][target.id] += current_prob * float(
                    transition_prob
                )

    return matrix_steps_states


def invert_2d_list(li: list[list[Any]]) -> list[list[Any]]:
    res = []
    for i in range(len(li[0])):
        sublist = []
        for j in range(len(li)):
            sublist.append(li[j][i])
        res.append(sublist)
    return res


def display_value_iteration_result(
    res: list[list[float]], hor_size: int, labels: list[str]
):
    """Display a value iteration results using matplotlib.

    Args:
        res (list[list[float]]): 2D list where result[n][m] is the probability to be in state m at step n.
        hor_size (int): the horizontal size of the plot, in inches.
        labels (list[str]): the names of all the states.
    """
    fig, ax = plt.subplots(1, 1)
    yticks = [s.labels[0] for s in labels] + [""]
    ax.set_xticks(range(len(res)))
    ax.set_yticks(range(len(res[0]) + 1))
    ax.set_yticklabels(yticks)

    ax.imshow(invert_2d_list(res), cmap="hot", interpolation="nearest", aspect="equal")
    plt.xlabel("steps")
    plt.ylabel("states")
    fig.set_size_inches(hor_size, hor_size)

    plt.show()
