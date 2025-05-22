import stormvogel.model
from stormvogel import parametric


def create_simple_pmc():
    # Create a new model with the name "simple pmc"
    pmc = stormvogel.model.new_dtmc("simple pmc")

    init = pmc.get_initial_state()

    # From the initial state, we have two transitions that either bring us further or to a sink state

    p1 = parametric.Polynomial()
    p2 = parametric.Polynomial()
    p1.set_coefficient((1, 1, 1), 4)
    p2.set_coefficient((1,), 1)
    p2.set_coefficient((0,), -1)

    pmc.new_state(labels=["goal"])
    pmc.new_state(labels=["sink"])

    init.set_transitions(
        [
            (p1, pmc.get_states_with_label("goal")[0]),
            (p2, pmc.get_states_with_label("sink")[0]),
        ]
    )

    # we add self loops to all states with no outgoing transitions
    pmc.add_self_loops()

    return pmc


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_simple_pmc().to_dot())
