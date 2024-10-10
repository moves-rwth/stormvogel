import stormpy
import stormpy.simulator
import stormvogel.result
import stormvogel.mapping
import stormvogel.model
import stormpy.examples.files
import stormpy.examples
import examples.die
import examples.monty_hall
import examples.monty_hall_pomdp
import random


class Path:
    """
    Path object that represents a path created by a simulator on a certain model.

    Args:
        path: The path itself is a dictionary where we either store for each step a state or a state action pair.
        model: The model the path traverses
    """

    path: (
        dict[int, tuple[stormvogel.model.Action, stormvogel.model.State]]
        | dict[int, stormvogel.model.State]
    )
    model: stormvogel.model.Model

    def __init__(
        self,
        path: dict[int, tuple[stormvogel.model.Action, stormvogel.model.State]]
        | dict[int, stormvogel.model.State],
        model: stormvogel.model.Model,
    ):
        if model.supports_rates():
            raise NotImplementedError
        self.path = path
        self.model = model

    def get_state_in_step(self, step: int) -> stormvogel.model.State | None:
        """returns the state discovered in the given step in the path"""
        if not self.model.supports_actions():
            state = self.path[step]
            assert isinstance(state, stormvogel.model.State)
            return state
        if self.supports_actions():
            t = self.path[step]
            assert (
                isinstance(t, tuple)
                and isinstance(t[0], stormvogel.model.Action)
                and isinstance(t[1], stormvogel.model.State)
            )
            state = t[1]
            assert isinstance(state, stormvogel.model.State)
            return state

    def get_action_in_step(self, step: int) -> stormvogel.model.Action | None:
        """returns the action discovered in the given step in the path"""
        if self.model.supports_actions():
            t = self.path[step]
            assert (
                isinstance(t, tuple)
                and isinstance(t[0], stormvogel.model.Action)
                and isinstance(t[1], stormvogel.model.State)
            )
            action = t[0]
            return action

    def get_step(
        self, step: int
    ) -> (
        tuple[stormvogel.model.Action, stormvogel.model.State] | stormvogel.model.State
    ):
        """returns the state or state action pair discovered in the given step"""
        return self.path[step]

    def __str__(self) -> str:
        path = "initial state"
        if self.model.supports_actions():
            for t in self.path.values():
                assert (
                    isinstance(t, tuple)
                    and isinstance(t[0], stormvogel.model.Action)
                    and isinstance(t[1], stormvogel.model.State)
                )
                path += f" --action: {t[0].name}--> state: {t[1].id}"
        else:
            for state in self.path.values():
                path += f" --> state: {state.id}"
        return path


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
) -> Path:
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

        # we start adding states or state action pairs to the path
        if not model.supports_actions():
            path = {}
            simulator.restart()
            for i in range(steps):
                state, reward, labels = simulator.step()
                path[i + 1] = model.states[state]
                if simulator.is_done():
                    break
        else:
            path = {}
            state, reward, labels = simulator.restart()
            for i in range(steps):
                actions = simulator.available_actions()
                select_action = (
                    random.randint(0, len(actions) - 1)
                    if not scheduler
                    else get_range_index(state)
                )
                new_action = actions[select_action]
                stormvogel_action = model.states[state].available_actions()[new_action]
                state, reward, labels = simulator.step(actions[select_action])
                path[i + 1] = (stormvogel_action, model.states[state])
                if simulator.is_done():
                    break
    else:
        raise NotImplementedError

    path_object = Path(path, model)

    return path_object


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
                        partial_model.new_action(
                            model.states[state].available_actions()[select_action].name
                        )
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
    """
    # we first test it with a dtmc
    dtmc = examples.die.create_die_dtmc()
    # rewardmodel = dtmc.add_rewards("rewardmodel")
    # for stateid in dtmc.states.keys():
    #    rewardmodel.rewards[stateid] = 5

    partial_model = simulate(dtmc, 1, 10)
    print(partial_model)
    path = simulate_path(dtmc, 5)
    print(path)

    """

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
    assert partial_model is not None
    print(partial_model.actions)
    # print(path)
    """
    # then we test it with a pomdp
    pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()
    # rewardmodel = pomdp.add_rewards("rewardmodel")
    # for i in range(67):
    #    rewardmodel.rewards[i] = 5

    taken_actions = {}
    for id, state in pomdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(pomdp, taken_actions)

    partial_model = simulate(pomdp, 10, 10, scheduler)
    path = simulate_path(pomdp, 5)
    print(partial_model)
    # print(partial_model.actions)
    print(path)
    """
