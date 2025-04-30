import stormvogel.model
from stormvogel import parametric


def create_die_dtmc():
    # Create a new model with the name "Die"
    dtmc = stormvogel.model.new_dtmc("Die")

    init = dtmc.get_initial_state()

    # From the initial state, we have two transitions that either bring us further or to a sink state

    p1 = parametric.Polynomial(1, 3)
    p2 = parametric.Polynomial(1, 1)
    p1.set_coefficient((1,1,1), 1)
    p2.set_coefficient((1,), 1)
    p2.set_coefficient((0,), -1)

    dtmc.new_state(labels=["goal"])
    dtmc.new_state(labels=["sink"])

    init.set_transitions(
        [
            (p1, dtmc.get_states_with_label("goal")[0]),
            (p2, dtmc.get_states_with_label("goal")[0]),
            (p2, dtmc.get_states_with_label("sink")[0]),
            (p1, dtmc.get_states_with_label("sink")[0]),
        ]
    )

    # we add self loops to all states with no outgoing transitions
    dtmc.add_self_loops()

    return dtmc


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_die_dtmc().to_dot())
