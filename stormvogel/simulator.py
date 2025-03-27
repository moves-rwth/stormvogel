import stormvogel.result
import stormvogel.mapping
import stormvogel.model
from typing import Callable
import random
from stormvogel.model import EmptyAction

try:
    import stormpy.simulator
except ImportError:
    stormpy = None


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
                path += f" --(action: {t[0].labels})--> state: {t[1].id}"
        else:
            for state in self.path.values():
                path += f" --> state: {state.id}"
        return path

    def __eq__(self, other):
        if isinstance(other, Path):
            if not self.model.supports_actions():
                return self.path == other.path and self.model == other.model
            else:
                if len(self.path) != len(other.path):
                    return False
                for tuple, other_tuple in zip(
                    sorted(self.path.values()), sorted(other.path.values())
                ):
                    assert not (
                        isinstance(tuple, stormvogel.model.State)
                        or isinstance(other_tuple, stormvogel.model.State)
                    )
                    if not (tuple[0] == other_tuple[0] and tuple[1] == other_tuple[1]):
                        return False
                return self.model == other.model
        else:
            return False


def get_range_index(
    state: stormvogel.model.State,
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action],
) -> int:
    """Helper function to convert the chosen action in a state by a scheduler to a range index."""
    assert scheduler is not None
    if isinstance(scheduler, stormvogel.result.Scheduler):
        action = scheduler.get_choice_of_state(state)
    elif callable(scheduler):
        action = scheduler(state)
    else:
        raise TypeError("Must be of type Scheduler or a function")

    available_actions = state.available_actions()

    assert action is not None
    return available_actions.index(action)


def simulate_path(
    model: stormvogel.model.Model,
    steps: int = 1,
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action]
    | None = None,
    seed: int | None = None,
) -> Path:
    """
    Simulates the model and returns the path created by the process.
    Args:
        model: The stormvogel model that the simulator should run on.
        steps: The number of steps the simulator walks through the model.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
                    (instead of a stormvogel scheduler, a function from states to actions can also be provided.)
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns a path object.
    """
    assert stormpy is not None

    # we initialize the simulator
    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    if seed:
        simulator = stormpy.simulator.create_simulator(stormpy_model, seed)
    else:
        simulator = stormpy.simulator.create_simulator(stormpy_model)
    assert simulator is not None

    # we start adding states or state action pairs to the path
    state_id = 0
    path = {}
    simulator.restart()
    if not model.supports_actions():
        for i in range(steps):
            # for each step we add a state to the path
            if not model.states[state_id].is_absorbing() and not simulator.is_done():
                state_id, reward, labels = simulator.step()
                path[i + 1] = model.states[state_id]
            else:
                break
    else:
        for i in range(steps):
            # we first choose an action (randomly or according to scheduler)
            actions = simulator.available_actions()
            select_action = (
                get_range_index(model.get_state_by_id(state_id), scheduler)
                if scheduler
                else random.randint(0, len(actions) - 1)
            )

            # we add the state action pair to the path
            stormvogel_action = model.states[state_id].available_actions()[
                select_action
            ]

            if (
                not model.states[state_id].is_absorbing(stormvogel_action)
                and not simulator.is_done()
            ):
                state_id, reward, labels = simulator.step(actions[select_action])
                path[i + 1] = (stormvogel_action, model.states[state_id])
            else:
                break

    path_object = Path(path, model)

    return path_object


def simulate(
    model: stormvogel.model.Model,
    steps: int = 1,
    runs: int = 1,
    scheduler: stormvogel.result.Scheduler
    | Callable[[stormvogel.model.State], stormvogel.model.Action]
    | None = None,
    seed: int | None = None,
) -> stormvogel.model.Model | None:
    """
    Simulates the model.
    Args:
        model: The stormvogel model that the simulator should run on
        steps: The number of steps the simulator walks through the model
        runs: The number of times the model gets simulated.
        scheduler: A stormvogel scheduler to determine what actions should be taken. Random if not provided.
                    (instead of a stormvogel scheduler, a function from states to actions can also be provided.)
        seed: The seed for the function that determines for each state what the next state will be. Random seed if not provided.

    Returns the partial model discovered by all the runs of the simulator together
    """
    assert stormpy is not None

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

            # we already set the rewards for the initial state/stateaction
            if model.supports_actions():
                try:
                    r = model.rewards[index].get_state_action_reward(
                        model.get_initial_state(), EmptyAction
                    )
                    assert r is not None
                    reward_model.set_state_action_reward(
                        partial_model.get_initial_state(),
                        EmptyAction,
                        r,
                    )
                except RuntimeError:
                    pass

            else:
                r = model.rewards[index].get_state_reward(model.get_initial_state())
                assert r is not None
                reward_model.set_state_reward(
                    partial_model.get_initial_state(),
                    r,
                )

    # now we start stepping through the model for the given number of runs
    discovered_states = {0}
    discovered_transitions = set()
    if not partial_model.supports_actions():
        for i in range(runs):
            simulator.restart()
            last_state_id = 0
            for j in range(steps):
                state_id, reward, labels = simulator.step()
                # we get the rewards in reversed order
                reward.reverse()

                # we add to the partial model what we discovered (if new)
                if state_id not in discovered_states:
                    discovered_states.add(state_id)
                    new_state = partial_model.new_state(
                        list(labels), name=str(state_id)
                    )

                    # we add the rewards
                    for index, rewardmodel in enumerate(partial_model.rewards):
                        rewardmodel.set_state_reward(new_state, reward[index])
                else:
                    new_state = partial_model.get_state_by_name(str(state_id))

                # we also add the transitions that we travelled through, so we need to keep track of the last state
                # and of the discovered transitions so that we don't add duplicates
                if (last_state_id, state_id) not in discovered_transitions:
                    probability = 0
                    transitions = model.get_transitions(last_state_id)
                    discovered_transitions.add((last_state_id, state_id))
                    for tuple in transitions.transition[
                        stormvogel.model.EmptyAction
                    ].branch:
                        if tuple[1].id == state_id:
                            probability += float(
                                tuple[0]
                            )  # if there are multiple transitions between the same pair of states, they collapse
                    assert new_state is not None
                    partial_model.get_state_by_name(str(last_state_id)).add_transitions(
                        [(probability, new_state)]
                    )

                last_state_id = state_id
                if simulator.is_done():
                    break
    else:
        discovered_actions = set()
        for i in range(runs):
            state_id = 0
            last_state_id = 0
            simulator.restart()
            for j in range(steps):
                # we first choose an action
                actions = simulator.available_actions()
                select_action = (
                    get_range_index(model.get_state_by_id(state_id), scheduler)
                    if scheduler
                    else random.randint(0, len(actions) - 1)
                )
                # we add the action to the partial model
                assert partial_model.actions is not None
                action = model.states[state_id].available_actions()[select_action]
                if action not in partial_model.actions:
                    partial_model.new_action(action.labels)

                # we get the new discovery
                discovery = simulator.step(actions[select_action])

                # we add the rewards.
                reward = discovery[1]
                for index, rewardmodel in enumerate(partial_model.rewards):
                    state = model.get_state_by_id(state_id)
                    rewardmodel.set_state_action_reward(state, action, reward[index])

                # we add the state to the model
                state_id, labels = discovery[0], discovery[2]
                if state_id not in discovered_states:
                    discovered_states.add(state_id)
                    new_state = partial_model.new_state(
                        list(labels), name=str(state_id)
                    )
                else:
                    new_state = partial_model.get_state_by_name(str(state_id))

                # we also add the transitions that we travelled through, so we need to keep track of the last state
                # and of the discovered transitions so that we don't add duplicates
                if (last_state_id, state_id, action) not in discovered_transitions:
                    probability = 0
                    transitions = model.get_state_by_id(last_state_id).get_outgoing_transitions(action)
                    discovered_transitions.add((last_state_id, state_id, action))
                    for tuple in transitions:
                        if tuple[1].id == state_id:
                            probability += float(
                                tuple[0]
                            )  # if there are multiple transitions between the same pair of action with next state, they collapse

                    assert new_state is not None
                    if (last_state_id, action) in discovered_actions:
                        i = partial_model.get_state_by_name(str(last_state_id)).id
                        branch = partial_model.transitions[i].transition[action]
                        branch.branch.append((probability, new_state))
                    else:
                        discovered_actions.add((last_state_id,action))
                        branch = stormvogel.model.Branch([(probability, new_state)])
                        trans = stormvogel.model.Transition({action: branch})
                        assert trans is not None
                        partial_model.get_state_by_name(str(last_state_id)).add_transitions(trans)

                last_state_id = state_id
                if simulator.is_done():
                    break

    return partial_model
