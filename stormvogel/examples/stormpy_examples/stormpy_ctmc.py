import numpy as np


def example_building_ctmcs_01():
    # Building the transition matrix using numpy
    import stormpy

    transitions = np.array(
        [
            [0, 1.5, 0, 0],
            [3, 0, 1.5, 0],
            [0, 3, 0, 1.5],
            [0, 0, 3, 0],
        ],
        dtype="float64",
    )

    # Default row groups: [0,1,2,3]
    transition_matrix = stormpy.build_sparse_matrix(transitions)
    # print(transition_matrix)

    # State labeling
    state_labeling = stormpy.storage.StateLabeling(4)
    state_labels = {"empty", "init", "deadlock", "full"}
    for label in state_labels:
        state_labeling.add_label(label)

    # Adding label to states
    state_labeling.add_label_to_state("init", 0)
    state_labeling.add_label_to_state("empty", 0)
    state_labeling.add_label_to_state("full", 3)

    # Collect components
    # rate_transitions = True, because the transition values are interpreted as rates
    components = stormpy.SparseModelComponents(
        transition_matrix=transition_matrix,
        state_labeling=state_labeling,
        rate_transitions=True,
    )

    # Build the model
    ctmc = stormpy.storage.SparseCtmc(components)

    return ctmc


if __name__ == "__main__":
    example_building_ctmcs_01()
