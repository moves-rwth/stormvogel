import stormpy.storage
import stormvogel.model


def simple_to_matrix(model: stormvogel.model.Model) -> stormpy.storage.SparseMatrix:
    """
    Takes a simple representation as input and outputs a sparsematrix
    """
    builder = stormpy.storage.SparseMatrixBuilder()

    for transition in model.transitions.items():
        for branch in transition[1].transition.values():
            for tuple in branch.branch:
                builder.add_next_value(
                    row=transition[0], column=tuple[1].id, value=tuple[0]
                )

    matrix = builder.build()
    return matrix


"""
def matrix_to_simple(matrix: stormpy.SparseMatrix) -> stormvogel.model.Model:

    Takes a sparsematrix as input and outputs a simple representation

    return simple
"""

if __name__ == "__main__":
    # Create a new model with the name "Die"
    dtmc = stormvogel.model.new_dtmc("Die")

    init = dtmc.get_initial_state()

    # From the initial state, add the transition to 6 new states with probability 1/6th.
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    print(simple_to_matrix(dtmc))
    "test"
