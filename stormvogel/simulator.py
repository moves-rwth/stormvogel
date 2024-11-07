import stormpy
import stormpy.simulator
import stormvogel.result
import stormvogel.mapping
import stormvogel.model
import stormpy.examples.files
import stormpy.examples
import random


class Path:
    """
    Path object that represents a path created by a simulator on a certain model.

    Args:
        path: The path itself is a dictionary where we either store for each step a state or a state action pair,
        depending on if we are working with a dtmc or an mdp.
        model: model that the path traverses through
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
        if model.get_type() != stormvogel.model.ModelType.MA:
            self.path = path
            self.model = model
        else:
            # TODO make the simulators work for markov automata
            raise NotImplementedError

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

    def __eq__(self, other):
        if isinstance(other, Path):
            return self.path == other.path and self.model == other.model
        else:
            return False


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: stormvogel.result.Scheduler | None = None,
    seed: int | None = None,
) -> Path:
    """
    Simulates the model and returns the path created by the process.
    Args:
        model: The stormvogel model that the simulator should run on.
        steps: The number of steps the simulator walks through the model.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns a path object.
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
    state = 0
    path = {}
    simulator.restart()
    if not model.supports_actions():
        for i in range(steps):
            # for each step we add a state to the path
            if not model.states[state].is_absorbing() and not simulator.is_done():
                state, reward, labels = simulator.step()
                path[i + 1] = model.states[state]
            else:
                break
    else:
        for i in range(steps):
            # we first choose an action (randomly or according to scheduler)
            actions = simulator.available_actions()
            select_action = (
                random.randint(0, len(actions) - 1)
                if not scheduler
                else get_range_index(state)
            )

            # we add the state action pair to the path
            stormvogel_action = model.states[state].available_actions()[select_action]

            if (
                not model.states[state].is_absorbing(stormvogel_action)
                and not simulator.is_done()
            ):
                state, reward, labels = simulator.step(actions[select_action])
                path[i + 1] = (stormvogel_action, model.states[state])
            else:
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
    Simulates the model.
    Args:
        model: The stormvogel model that the simulator should run on
        steps: The number of steps the simulator walks through the model
        runs: The number of times the model gets simulated.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns the partial model discovered by all the runs of the simulator together
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
            last_state = 0
            for j in range(steps):
                state, reward, labels = simulator.step()
                reward.reverse()

                # we add to the partial model what we discovered (if new)
                if state not in discovered_states:
                    discovered_states.add(state)

                    # we also add the transitions that we travelled through, so we need to keep track of the last state
                    probability = 0
                    transitions = model.get_transitions(last_state)
                    for tuple in transitions.transition[
                        stormvogel.model.EmptyAction
                    ].branch:
                        if tuple[1].id == state:
                            probability += float(tuple[0])

                    new_state = partial_model.new_state(list(labels))
                    partial_model.get_state_by_id(last_state).add_transitions(
                        [(probability, new_state)]
                    )

                    for index, rewardmodel in enumerate(partial_model.rewards):
                        rewardmodel.set(new_state, reward[index])

                    last_state = state
                if simulator.is_done():
                    break
    else:
        state = 0
        last_state_partial = partial_model.get_initial_state()
        last_state_id = 0
        for i in range(runs):
            simulator.restart()
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
                action = model.states[state].available_actions()[select_action]
                if action not in partial_model.actions.values():
                    partial_model.new_action(action.name)

                # we add the reward model to the partial model
                discovery = simulator.step(actions[select_action])
                reward = discovery[1]
                for index, rewardmodel in enumerate(partial_model.rewards):
                    row_group = stormpy_model.transition_matrix.get_row_group_start(
                        state
                    )
                    state_action_pair = row_group + select_action
                    rewardmodel.set_action_state(state_action_pair, reward[index])

                # we add the state
                state, labels = discovery[0], discovery[2]
                if state not in discovered_states:
                    discovered_states.add(state)

                    # we also add the transitions that we travelled through, so we need to keep track of the last state
                    probability = 0
                    transitions = model.get_transitions(last_state_id)
                    for tuple in transitions.transition[action].branch:
                        if tuple[1].id == state:
                            probability += float(tuple[0])

                    new_state = partial_model.new_state(list(labels))
                    last_state_partial.add_transitions([(probability, new_state)])

                    last_state_partial = new_state
                    last_state_id = state
                if simulator.is_done():
                    break

    return partial_model
