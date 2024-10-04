import stormvogel.model


def create_die_dtmc():
    # Create a new model with the name "Die"
    dtmc = stormvogel.model.new_dtmc("Die")

    init = dtmc.get_initial_state()

    # From the initial state, add the transition to 6 new states with probability 1/6th.
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    # we add self loops to all states with no outgoing transitions
    dtmc.add_self_loops()

    # test if state deletion works
    # dtmc.delete_state(dtmc.get_state_by_id(1), True, True)

    return dtmc


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_die_dtmc().to_dot())
