import stormpy.storage
import stormvogel.model


def stormvogel_to_stormpy_dtmc(
    model: stormvogel.model.Model,
) -> stormpy.storage.SparseDtmc:
    """
    Takes a simple representation as input and outputs a dtmc how it is represented in stormpy
    """

    # we first build the SparseMatrix
    builder = stormpy.storage.SparseMatrixBuilder()
    for transition in model.transitions.items():
        for branch in transition[1].transition.values():
            for tuple in branch.branch:
                builder.add_next_value(
                    row=transition[0], column=tuple[1].id, value=tuple[0]
                )
    matrix = builder.build()

    # then we add the labels
    state_labeling = stormpy.storage.StateLabeling(len(model.states))
    for state in model.states.items():
        for label in state[1].labels:
            state_labeling.add_label(label)
            state_labeling.add_label_to_state(label, state[0])

    # TODO rewards

    # then we build the dtmc
    components = stormpy.SparseModelComponents(
        transition_matrix=matrix, state_labeling=state_labeling
    )
    dtmc = stormpy.storage.SparseDtmc(components)

    return dtmc


def stormpy_to_stormvogel_dtmc(
    sparsedtmc: stormpy.storage.SparseDtmc,
) -> stormvogel.model.Model:
    """
    Takes a dtmc stormpy representation as input and outputs a simple stormvogel representation
    """

    # we create the model
    model = stormvogel.model.new_dtmc(name=None)

    # we add the states and the transitions
    matrix = sparsedtmc.transition_matrix
    for state in sparsedtmc.states:
        # the initial state is automatically added so we don't add it
        if state.id > 0:
            model.new_state(labels=list(state.labels))

        # we add the transitions
        row = matrix.get_row(state.id)
        transitionshorthand = [(x.value(), x.column) for x in row]
        transitions = stormvogel.model.transition_from_shorthand(transitionshorthand)
        model.set_transitions(state, transitions)

    # TODO rewards

    return model


"""
def stormvogel_to_stormpy_mdp(model: stormvogel.model.Model) -> stormpy.storage.SparseMdp:
    Takes a simple representation as input and outputs an mdp how it is represented in stormpy

"""


"""
def stormpy_to_stormvogel_mdp(matrix: stormpy.SparseMatrix) -> stormvogel.model.Model:
    Takes a stormpy representation of an mdp as input and outputs a stormvogel model representation
"""


if __name__ == "__main__":
    dtmc = stormvogel.model.new_dtmc("Die")
    init = dtmc.get_initial_state()
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )
    print(dtmc)
    sparsedtmc = stormvogel_to_stormpy_dtmc(dtmc)
    print(stormpy_to_stormvogel_dtmc(sparsedtmc))
