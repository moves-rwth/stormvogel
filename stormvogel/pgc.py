import stormvogel.model
from dataclasses import dataclass
from collections.abc import Callable


@dataclass
class Action:
    """pgc action object. Contains a list of labels"""

    labels: list[str]


@dataclass
class State:
    """pgc state object. Can contain any number of any type of arguments"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"State({self.__dict__})"

    def __hash__(self):
        return hash(self.__dict__)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__dict__ == other.__dict__
        return False


def valid_input(
    delta,
    initial_state_pgc,
    rewards,
    labels,
    available_actions,
    modeltype,
    max_size=2000,
):
    """
    function that checks if the input for the pgc model builder is valid
    it will give a runtime error if it isn't.
    """

    if modeltype == stormvogel.model.ModelType.MDP and available_actions is None:
        raise RuntimeError(
            "You have to provide an available actions function for models that support actions"
        )

    supports_actions = modeltype in (
        stormvogel.model.ModelType.MDP,
        stormvogel.model.ModelType.POMDP,
        stormvogel.model.ModelType.MA,
    )

    # now we simulate the behaviour of the pgc model builder and provide the necessary errors
    states_seen = []
    states_to_be_visited = [initial_state_pgc]
    state = initial_state_pgc
    while len(states_to_be_visited) > 0:
        state = states_to_be_visited[0]
        states_to_be_visited.remove(state)
        if supports_actions:
            assert available_actions is not None
            actionslist = available_actions(state)
            if not isinstance(actionslist, list):
                raise RuntimeError(
                    f"On input {state}, the available actions function does not return a list. Make sure to change it to [<your output>]"
                )
            for action in actionslist:
                tuples = delta(state, action)

                if not isinstance(tuples, list):
                    raise RuntimeError(
                        f"On input pair {state} {action}, the delta function does not return a list. Make sure to change it to [<your output>]"
                    )

                for tuple in tuples:
                    if tuple[1] not in states_seen:
                        states_seen.append(tuple[1])
                        states_to_be_visited.append(tuple[1])
        else:
            tuples = delta(state)
            if not isinstance(tuples, list):
                raise RuntimeError(
                    f"On input {state}, the delta function does not return a list. Make sure to change it to [<your output>]"
                )

            for tuple in tuples:
                if tuple[1] not in states_seen:
                    states_seen.append(tuple[1])
                    states_to_be_visited.append(tuple[1])

        # if at some point we discovered more than max_size states, we complain
        if len(states_seen) > max_size:
            raise RuntimeError(
                f"The model you want te create has a very large amount of states (at least {max_size}), if you wish to proceed, set max_size to some larger number."
            )

    # we check for the rewards
    if rewards is not None:
        if supports_actions:
            assert available_actions is not None
            nr = len(
                rewards(initial_state_pgc, available_actions(initial_state_pgc)[0])
            )

            for state in states_seen:
                assert available_actions is not None
                for action in available_actions(state):
                    rewardlist = rewards(state, action)
                    if not isinstance(rewardlist, list):
                        raise RuntimeError(
                            f"On input pair {state} {action}, the rewards function does not return a list. Make sure to change it to [<your output>]"
                        )
                    if len(rewardlist) != nr:
                        raise RuntimeError(
                            "Make sure that the rewards function returns a list with the same length on each return"
                        )

        else:
            nr = len(rewards(initial_state_pgc))

            for state in states_seen:
                rewardlist = rewards(state)
                if not isinstance(rewardlist, list):
                    raise RuntimeError(
                        f"On input {state}, the rewards function does not return a list. Make sure to change it to [<your output>]"
                    )
                if len(rewardlist) != nr:
                    raise RuntimeError(
                        "Make sure that the rewards function returns a list with the same length on each return"
                    )

    # we check for the labels
    if labels is not None:
        for state in states_seen:
            labellist = labels(state)
            if not isinstance(labellist, list):
                raise RuntimeError(
                    f"On input {state}, the labels function does not return a list. Make sure to change it to [<your output>]"
                )


def build_pgc(
    delta: Callable,
    initial_state_pgc,
    rewards: Callable | None = None,
    labels: Callable | None = None,
    available_actions: Callable | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
    max_size: int = 2000,
    check_validity: bool = True,
) -> stormvogel.model.Model:
    """
    function that converts a delta function, an available_actions function an initial state and a model type
    to a stormvogel model

    this works analogous to a prism file, where the delta is the module in this case.

    (this function uses the pgc classes State and Action instead of the ones from stormvogel.model)
    """

    if check_validity:
        valid_input(
            delta,
            initial_state_pgc,
            rewards,
            labels,
            available_actions,
            modeltype,
            max_size,
        )

    # we create the model with the given type and initial state
    model = stormvogel.model.new_model(modeltype=modeltype, create_initial_state=False)
    model.new_state(labels=["init", str(initial_state_pgc)])

    # we continue calling delta and adding new states until no states are
    # left to be checked
    states_seen = []
    states_to_be_visited = [initial_state_pgc]
    while len(states_to_be_visited) > 0:
        state = states_to_be_visited[0]
        states_to_be_visited.remove(state)
        transition = {}

        if state not in states_seen:
            states_seen.append(state)

        if model.supports_actions():
            # we loop over all available actions and call the delta function for each actions
            assert available_actions is not None

            for action in available_actions(state):
                try:
                    if action.labels != []:
                        stormvogel_action = model.new_action(
                            frozenset(
                                action.labels  # type: ignore
                            ),  # right now we only look at one label
                        )
                    else:
                        stormvogel_action = stormvogel.model.EmptyAction
                except RuntimeError:
                    if action.labels != []:
                        stormvogel_action = model.get_action_with_labels(
                            frozenset(action.labels)
                        )
                    else:
                        stormvogel_action = stormvogel.model.EmptyAction

                tuples = delta(state, action)

                # we add all the newly found transitions to the model
                branch = []
                for tuple in tuples:
                    if tuple[1] not in states_seen:
                        states_seen.append(tuple[1])
                        new_state = model.new_state(labels=str(tuple[1]))
                        branch.append((tuple[0], new_state))
                        states_to_be_visited.append(tuple[1])
                    else:
                        branch.append(
                            (tuple[0], model.get_states_with_label(str(tuple[1]))[0])
                        )
                if branch != []:
                    transition[stormvogel_action] = stormvogel.model.Branch(branch)
        else:
            tuples = delta(state)
            # we add all the newly found transitions to the model
            branch = []
            for tuple in tuples:
                if tuple[1] not in states_seen:
                    states_seen.append(tuple[1])
                    new_state = model.new_state(labels=str(tuple[1]))

                    branch.append((tuple[0], new_state))
                    states_to_be_visited.append(tuple[1])
                else:
                    branch.append(
                        (tuple[0], model.get_states_with_label(str(tuple[1]))[0])
                    )
                if branch != []:
                    transition[stormvogel.model.EmptyAction] = stormvogel.model.Branch(
                        branch
                    )
        s = model.get_states_with_label(str(state))[0]
        assert s is not None
        model.add_transitions(
            s,
            stormvogel.model.Transition(transition),
        )

    # we add the rewards
    if rewards is not None:
        if model.supports_actions():
            # we first create the right number of reward models
            assert available_actions is not None
            nr = len(
                rewards(initial_state_pgc, available_actions(initial_state_pgc)[0])
            )
            for i in range(nr):
                model.add_rewards("rewardmodel: " + str(i))

            for state in states_seen:
                assert available_actions is not None
                for action in available_actions(state):
                    rewardlist = rewards(state, action)
                    s = model.get_states_with_label(str(state))[0]
                    assert s is not None
                    for index, reward in enumerate(rewardlist):
                        a = model.get_action_with_labels(frozenset(action.labels))
                        assert a is not None
                        model.rewards[index].set_state_action_reward(
                            s,
                            a,
                            reward,
                        )
        else:
            # we first create the right number of reward models
            nr = len(rewards(initial_state_pgc))
            for i in range(nr):
                model.add_rewards("rewardmodel: " + str(i))

            for state in states_seen:
                rewardlist = rewards(state)
                s = model.get_states_with_label(str(state))[0]
                assert s is not None
                for index, reward in enumerate(rewardlist):
                    model.rewards[index].set_state_reward(s, reward)

    # we add the labels
    if labels is not None:
        for state in states_seen:
            s = model.get_states_with_label(str(state))[0]
            if "init" in s.labels:
                s.labels = ["init"]
            else:
                s.labels = []
            assert s is not None
            for label in labels(state):
                s.add_label(label)
    else:
        for state in states_seen:
            s = model.get_states_with_label(str(state))[0]
            if "init" in s.labels:
                s.labels = ["init"]
            else:
                s.labels = []

    return model
