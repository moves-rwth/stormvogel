"""Some example applications of the stormvogel API!"""

from typing import Any
import stormvogel.model
import matplotlib.pyplot as plt


def naive_value_iteration(
    model: stormvogel.model.Model, epsilon: float, target_state: stormvogel.model.State
) -> list[list[stormvogel.model.Number]]:
    """Run naive value iteration. The result is a 2D list where result[n][m] is the probability to be in state m at step n.

    Args:
        model (stormvogel.model.Model): Target model.
        steps (int): Amount of steps.
        target_state (stormvogel.model.State): Target state of the model.

    Returns:
        list[list[float]]: The result is a 2D list where result[n][m] is the value of state m at iteration n.
    """
    if epsilon <= 0:
        RuntimeError("The algorithm will not terminate if epsilon is zero.")

    # Create a dynamic matrix (list of lists) to store the result.
    values_matrix = [[0 for state in model.get_states()]]
    values_matrix[0][target_state.id] = 1

    terminate = False
    while not terminate:
        old_values = values_matrix[len(values_matrix) - 1]
        new_values = [None for state in model.get_states()]
        for sid, state in model.get_states().items():
            transitions = model.get_transitions(state)
            # Now we have to take a decision for an action.
            action_values = {}
            for action, branch in transitions.transition.items():
                branch_value = sum(
                    [prob * old_values[state.id] for (prob, state) in branch.branch]  # type: ignore
                )
                action_values[action] = branch_value
            # We take the action with the highest value.
            highest_value = max(action_values.values())
            new_values[sid] = highest_value
        values_matrix.append(new_values)  # type: ignore
        terminate = (
            sum([abs(x - y) for (x, y) in zip(new_values, old_values)]) < epsilon  # type: ignore
        )
    return values_matrix  # type: ignore


def dtmc_evolution(model: stormvogel.model.Model, steps: int) -> list[list[float]]:
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
    matrix_steps_states[0][model.get_initial_state().id] = 1

    # Apply the updated values for each step.
    for current_step in range(steps - 1):
        next_step = current_step + 1
        for s_id, s in model.get_states().items():
            branch = model.get_branch(s)
            for transition_prob, target in branch.branch:
                current_prob = matrix_steps_states[current_step][s_id]
                assert isinstance(transition_prob, (int, float))
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
    ax.set_xticks(range(len(res)))
    ax.set_yticks(range(len(res[0]) + 1))
    ax.set_yticklabels(labels + [""])

    ax.imshow(invert_2d_list(res), cmap="hot", interpolation="nearest", aspect="equal")
    plt.xlabel("steps")
    plt.ylabel("states")
    fig.set_size_inches(hor_size, hor_size)

    plt.show()
