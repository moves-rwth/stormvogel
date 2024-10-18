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
import examples.nuclear_fusion_ctmc
import examples.simple_ma
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
                path += f" --(action: {t[0].name})--> state: {t[1].id}"
        else:
            for state in self.path.values():
                path += f" --> state: {state.id}"
        return path


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
    seed: int | None = None,
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

    # we initialize the simulator
    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    if seed:
        simulator = stormpy.simulator.create_simulator(stormpy_model, seed)
    else:
        simulator = stormpy.simulator.create_simulator(stormpy_model)
    assert simulator is not None

    # we start adding states or state action pairs to the path
    if not model.supports_actions():
        path = {}
        simulator.restart()
        for i in range(steps):
            # for each step we add a state to the path
            state, reward, labels = simulator.step()
            path[i + 1] = model.states[state]
            if simulator.is_done():
                break
    else:
        if model.get_type() == stormvogel.model.ModelType.POMDP:
            simulator.set_full_observability(True)
        path = {}
        state, reward, labels = simulator.restart()
        for i in range(steps):
            # we first choose an action (randomly or according to scheduler)
            actions = simulator.available_actions()
            select_action = (
                random.randint(0, len(actions) - 1)
                if not scheduler
                else get_range_index(state)
            )

            # we add the state action pair to the path
            stormvogel_action = stormvogel.model.EmptyAction
            next_step = simulator.step(actions[select_action])
            state, reward, labels = next_step
            path[i + 1] = (stormvogel_action, model.states[state])
            if simulator.is_done():
                break

    path_object = Path(path, model)

    return path_object


def simulate(
    model: stormvogel.model.Model,
    steps: int = 1,
    runs: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
    seed: int | None = None,
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

    # we initialize the simulator
    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    assert stormpy_model is not None
    if seed:
        simulator = stormpy.simulator.create_simulator(stormpy_model, seed)
    else:
        simulator = stormpy.simulator.create_simulator(stormpy_model)
    assert simulator is not None

    # we keep track of all discovered states over all runs and add them to the partial model
    # we also add the discovered rewards and actions to the partial model if present
    partial_model = stormvogel.model.new_model(model.get_type())

    # we add each rewardmodel to the partial model
    if model.rewards:
        for index, reward in enumerate(model.rewards):
            reward_model = partial_model.add_rewards(model.rewards[index].name)

            # we already set the rewards for the initial state
            reward_model.set(
                partial_model.get_initial_state(),
                model.rewards[index].get(model.get_initial_state()),
            )

    # now we start stepping through the model
    discovered_states = {0}
    if not partial_model.supports_actions():
        for i in range(runs):
            simulator.restart()
            for j in range(steps):
                state, reward, labels = simulator.step()
                reward.reverse()

                # we add to the partial model what we discovered (if new)
                if state not in discovered_states:
                    discovered_states.add(state)
                    new_state = partial_model.new_state(list(labels))
                    for index, rewardmodel in enumerate(partial_model.rewards):
                        rewardmodel.set(new_state, reward[index])

                if simulator.is_done():
                    break
    else:
        for i in range(runs):
            state, reward, labels = simulator.restart()
            for j in range(steps):
                # we first choose an action

                actions = simulator.available_actions()
                select_action = (
                    random.randint(0, len(actions) - 1)
                    if not scheduler
                    else get_range_index(state)
                )

                # we add the action to the partial model
                assert partial_model.actions is not None
                if (
                    model.states[state].available_actions()[select_action]
                    not in partial_model.actions.values()
                ):
                    partial_model.new_action(
                        model.states[state].available_actions()[select_action].name
                    )

                # we add the other discoveries to the partial model
                discovery = simulator.step(actions[select_action])
                reward = discovery[1]
                for index, rewardmodel in enumerate(partial_model.rewards):
                    row_group = stormpy_model.transition_matrix.get_row_group_start(
                        state
                    )
                    state_action_pair = row_group + select_action
                    rewardmodel.set_action_state(state_action_pair, reward[index])
                state, labels = discovery[0], discovery[2]
                if state not in discovered_states:
                    discovered_states.add(state)
                    partial_model.new_state(list(labels))

                if simulator.is_done():
                    break

    return partial_model


if __name__ == "__main__":
    """
    # we first test it with a dtmc
    dtmc = examples.die.create_die_dtmc()
    rewardmodel = dtmc.add_rewards("rewardmodel")
    for stateid in dtmc.states.keys():
        rewardmodel.rewards[stateid] = 1

    rewardmodel2 = dtmc.add_rewards("rewardmodel2")
    for stateid in dtmc.states.keys():
        rewardmodel2.rewards[stateid] = 2

    partial_model = simulate(dtmc, 1, 10)
    print(partial_model)
    path = simulate_path(dtmc, 5)
    print(path)
    """

    # then we test it with an mdp
    mdp = examples.monty_hall.create_monty_hall_mdp()
    rewardmodel = mdp.add_rewards("rewardmodel")
    for i in range(67):
        rewardmodel.rewards[i] = i
    rewardmodel2 = mdp.add_rewards("rewardmodel2")
    for i in range(67):
        rewardmodel2.rewards[i] = i

    taken_actions = {}
    for id, state in mdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(mdp, taken_actions)

    partial_model = simulate(mdp, 10, 1, scheduler)
    path = simulate_path(mdp, 5)
    print(partial_model)
    assert partial_model is not None
    print(path)
    print(partial_model.rewards)

    """
    # then we test it with a pomdp
    pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()

    taken_actions = {}
    for id, state in pomdp.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(pomdp, taken_actions)

    partial_model = simulate(pomdp, 10, 10, scheduler)
    path = simulate_path(pomdp, 5)
    print(partial_model)
    print(path)

    # then we test it with a ctmc
    ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    partial_model = simulate(ctmc, 10, 10)
    path = simulate_path(ctmc, 5)
    print(partial_model)
    print(path)

    # TODO Markov automatas


    ma = examples.simple_ma.create_simple_ma()

    taken_actions = {}
    for id, state in ma.states.items():
        taken_actions[id] = state.available_actions()[0]
    scheduler = stormvogel.result.Scheduler(ma, taken_actions)

    partial_model = simulate(ma, 10, 10, scheduler)
    #path = simulate_path(ma, 5)
    print(partial_model)
    #print(path)
    """
