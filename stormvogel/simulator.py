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
    model: stormvogel.model.Model,
    steps: int = 1,
    runs: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
) -> stormvogel.model.Model | None:
    """
    Simulates the model a given number of steps for a given number of runs.
    Returns the partial model discovered by the simulator
    """

    def get_range_index(stateid: int):
        """Helper function to convert the chosen action in a state by a scheduler to a range index."""
        assert scheduler is not None
        action = scheduler.get_choice_of_state(model.get_state_by_id(state))
        available_actions = model.states[stateid].available_actions()

        assert action is not None
        return available_actions.index(action)

    if not model.supports_rates():
        # we initialize the simulator
        stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
        simulator = stormpy.simulator.create_simulator(stormpy_model)
        assert simulator is not None

        # we keep track of all discovered states over all runs and add them to the partial model
        partial_model = stormvogel.model.new_model(model.get_type())
        if not partial_model.supports_actions():
            for i in range(runs):
                simulator.restart()
                for j in range(steps):
                    state, reward, labels = simulator.step()
                    if state not in partial_model.states.keys():
                        partial_model.new_state(list(labels))
                    if simulator.is_done():
                        break
        else:
            for i in range(runs):
                state, reward, labels = simulator.restart()
                for j in range(steps):
                    actions = simulator.available_actions()
                    select_action = (
                        random.randint(0, len(actions) - 1)
                        if not scheduler
                        else get_range_index(state)
                    )
                    state, reward, labels = simulator.step(actions[select_action])
                    if state not in partial_model.states.keys():
                        partial_model.new_state(list(labels))
                    if simulator.is_done():
                        break
    else:
        raise NotImplementedError

    return partial_model


if __name__ == "__main__":
    dtmc = examples.die.create_die_dtmc()
    partial_model = simulate(dtmc, 1, 10)

    print(partial_model)

    mdp = examples.monty_hall.create_monty_hall_mdp()
    taken_actions = {}
    for id, state in mdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = simulate(mdp, 3, 1, scheduler)

    print(partial_model)
