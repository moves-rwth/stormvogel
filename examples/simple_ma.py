import stormvogel.model


def create_simple_ma():
    # Create a new model
    ma = stormvogel.model.new_ma("example ma")

    init = ma.get_initial_state()

    # We have 2 actions
    init.set_transitions(
        [
            (
                ma.action(f"{i}"),
                ma.new_state(),
            )
            for i in range(5)
        ]
    )

    # We add the rates and markovian states
    ma.markovian_states = [0, 3, 4]
    ma.exit_rates = {i: 0 for i in range(6)}

    return ma


if __name__ == "__main__":
    # Print the resulting model in dot format.
    print(create_simple_ma().to_dot())