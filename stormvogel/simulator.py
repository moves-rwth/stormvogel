import stormpy
import stormpy.simulator
import stormvogel.result
import stormvogel.mapping
import stormvogel.model
import stormpy.examples.files
import stormpy.examples
import examples.die
import examples.monty_hall
import random


def simulate(
    model: stormvogel.model.Model, steps: int = 1, runs: int = 1
) -> stormvogel.model.Model | None:
    """
    Simulates the model a given number of steps for a given number of runs.
    Returns the partial model discovered by the simulator
    """

    if not model.supports_rates():
        stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
        simulator = stormpy.simulator.create_simulator(stormpy_model)
        partial_model = stormvogel.model.new_model(model.get_type())

        assert simulator is not None

        if not partial_model.supports_actions():
            discovered_states = set()
            for i in range(runs):
                simulator.restart()
                for j in range(steps):
                    state, reward, labels = simulator.step()
                    if state not in discovered_states:
                        partial_model.new_state(list(labels))
                        discovered_states.add(state)
                    if simulator.is_done():
                        break
        elif partial_model.supports_actions():
            discovered_states = set()
            # discovered_actions = set()
            for i in range(runs):
                simulator.restart()
                for j in range(steps):
                    actions = simulator.available_actions()
                    select_action = random.randint(0, len(actions) - 1)
                    # discovered_actions.add(select_action)
                    state, reward, labels = simulator.step(actions[select_action])
                    if state not in discovered_states:
                        partial_model.new_state(list(labels))
                    discovered_states.add(state)
                    if simulator.is_done():
                        break
    else:
        raise NotImplementedError

    return partial_model


if __name__ == "__main__":
    # dtmc = examples.die.create_die_dtmc()
    # partial_model = simulate(dtmc, 1, 10)

    mdp = examples.monty_hall.create_monty_hall_mdp()
    partial_model = simulate(mdp, 100, 100)

    print(partial_model)
