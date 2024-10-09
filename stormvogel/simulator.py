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


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
) -> str | None:
    """
    Simulates the model a given number of steps.
    Returns the resulting path of the simulator.
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

        path = f"{0}"
        if not model.supports_actions():
            simulator.restart()
            for j in range(steps):
                state, reward, labels = simulator.step()
                path += f" --> {state}"
                if simulator.is_done():
                    break
        else:
            state, reward, labels = simulator.restart()
            for j in range(steps):
                actions = simulator.available_actions()
                select_action = (
                    random.randint(0, len(actions) - 1)
                    if not scheduler
                    else get_range_index(state)
                )
                path += f" --act={actions[select_action]}--> "
                state, reward, labels = simulator.step(actions[select_action])
                path += f"{state}"
                if simulator.is_done():
                    break
    else:
        raise NotImplementedError

    return path


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
        # we also add the discovered rewards and actions to the partial model if present
        partial_model = stormvogel.model.new_model(model.get_type())
        reward_model = partial_model.add_rewards("rewards")
        if not partial_model.supports_actions():
            for i in range(runs):
                simulator.restart()
                for j in range(steps):
                    state, reward, labels = simulator.step()
                    if state not in partial_model.states.keys():
                        partial_model.new_state(list(labels))
                        reward_model.rewards[state] = reward
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
                    assert partial_model.actions is not None
                    if (
                        model.states[state].available_actions()[select_action]
                        not in partial_model.actions.values()
                    ):
                        partial_model.new_action(str(i) + str(j))
                    state, reward, labels = simulator.step(actions[select_action])
                    if state not in partial_model.states.keys():
                        partial_model.new_state(list(labels))
                        reward_model.rewards[state] = reward
                    if simulator.is_done():
                        break
    else:
        raise NotImplementedError

    return partial_model


if __name__ == "__main__":
    # we first test it with a dtmc
    dtmc = examples.die.create_die_dtmc()
    # rewardmodel = dtmc.add_rewards("rewardmodel")
    # for stateid in dtmc.states.keys():
    #    rewardmodel.rewards[stateid] = 5

    partial_model = simulate(dtmc, 1, 10)
    print(partial_model)
    path = simulate_path(dtmc, 5)
    print(path)

    # then we test it with an mdp
    mdp = examples.monty_hall.create_monty_hall_mdp()
    # rewardmodel = mdp.add_rewards("rewardmodel")
    # for i in range(67):
    #    rewardmodel.rewards[i] = 5

    taken_actions = {}
    for id, state in mdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = simulate(mdp, 10, 10, scheduler)
    path = simulate_path(mdp, 5)
    print(partial_model)
    # print(partial_model.actions)
    print(path)
