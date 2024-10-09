import stormpy
import stormpy.simulator
import stormvogel.result
import stormvogel.mapping
import stormvogel.model
import stormpy.examples.files
import stormpy.examples
import examples.die


def simulate(
    model: stormvogel.model.Model, steps: int = 1, runs: int = 1
) -> stormvogel.model.Model:
    """
    Simulates the model a given number of steps for a given number of runs.
    Returns the partial model discovered by the simulator
    """

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    simulator = stormpy.simulator.create_simulator(stormpy_model)
    partial_model = stormvogel.model.new_model(model.get_type())

    assert simulator is not None
    discovered_states = []
    for i in range(runs):
        simulator.restart()
        for j in range(steps):
            state, reward, labels = simulator.step()
            if state not in discovered_states:
                partial_model.new_state(list(labels))
            discovered_states.append(state)
            if simulator.is_done():
                break

    return partial_model


if __name__ == "__main__":
    dtmc = examples.die.create_die_dtmc()

    partial_model = simulate(dtmc, 1, 100)

    print(partial_model)
