import stormvogel.model
from dataclasses import dataclass
from collections.abc import Callable
import inspect


@dataclass
class Action:
    """pgc action object. Contains a list of labels"""

    labels: list[str]


PgcEmpytAction = Action([])


@dataclass
class State:
    """pgc state object. Can contain any number of any type of arguments"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"State({self.__dict__})"

    def __hash__(self):
        return hash(str(self.__dict__))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__dict__ == other.__dict__
        return False


def valid_input(
    delta: Callable,
    initial_state_pgc,
    rewards: Callable | None = None,
    labels: Callable | None = None,
    available_actions: Callable | None = None,
    observations: Callable | None = None,
    rates: Callable | None = None,
    valuations: Callable | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
):
    """
    function that checks if the input for the pgc model builder is valid
    it will give a runtime error if it isn't.
    """

    supports_actions = modeltype in (
        stormvogel.model.ModelType.MDP,
        stormvogel.model.ModelType.POMDP,
        stormvogel.model.ModelType.MA,
    )

    # we first check if we have an available_actions function in case our model supports actions
    if supports_actions and available_actions is None:
        raise ValueError(
            "You have to provide an available actions function for models that support actions"
        )

    # and we check if we have an observations function in case our model is a POMDP
    if modeltype == stormvogel.model.ModelType.POMDP and observations is None:
        raise ValueError("You have to provide an observations function for pomdps")

    # we check if the provided functions have the right number of parameters
    if supports_actions:
        assert available_actions is not None
        sig = inspect.signature(available_actions)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The available_actions function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    sig = inspect.signature(delta)
    num_params = len(sig.parameters)
    if supports_actions:
        if num_params != 2:
            raise ValueError(
                f"Your delta function must take exactly two arguments (state and action), but it takes {num_params} arguments"
            )
    else:
        if num_params != 1:
            raise ValueError(
                f"Your delta function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if rewards is not None:
        sig = inspect.signature(rewards)
        num_params = len(sig.parameters)
        if supports_actions:
            if num_params != 2:
                raise ValueError(
                    f"The rewards function must take exactly two arguments (state, action), but it takes {num_params} arguments"
                )
        else:
            if num_params != 1:
                raise ValueError(
                    f"The rewards function must take exactly one argument (state), but it takes {num_params} arguments"
                )

    if labels is not None:
        sig = inspect.signature(labels)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The labels function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if observations is not None:
        sig = inspect.signature(observations)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The observations function must take exactly one argument (state), but it takes {num_params} arguments"
            )

    if valuations is not None:
        sig = inspect.signature(valuations)
        num_params = len(sig.parameters)
        if num_params != 1:
            raise ValueError(
                f"The valuations function must take exactly one argument (state), but it takes {num_params} arguments"
            )


def build_pgc(
    delta: Callable,
    initial_state_pgc,
    rewards: Callable | None = None,
    labels: Callable | None = None,
    available_actions: Callable | None = None,
    observations: Callable | None = None,
    rates: Callable | None = None,
    valuations: Callable | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
    max_size: int = 2000,
) -> stormvogel.model.Model:
    """
    function that converts a delta function, an available_actions function an initial state and a model type
    to a stormvogel model

    this works analogous to a prism file, where the delta is the module in this case.

    (this function uses the pgc classes State and Action instead of the ones from stormvogel.model)
    """

    def add_new_transitions(tuples, states_seen, state):
        """
        helper function to add all the newly found transitions to the model
        """
        branch = []
        if tuples is not None:
            for tup in tuples:
                # in case only a state is provided, we assume the probability is 1
                if not isinstance(tup, tuple):
                    s = tup
                    val = 1
                else:
                    s = tup[1]
                    val = tup[0]

                if s not in states_seen:
                    states_seen.append(s)
                    new_state = model.new_state()
                    state_lookup[s] = new_state
                    branch.append((val, new_state))
                    states_to_be_visited.append(s)
                else:
                    branch.append((val, state_lookup[s]))
        else:
            # if we have no return value, we add a self loop
            branch.append((1, state_lookup[state]))
        return branch

    valid_input(
        delta,
        initial_state_pgc,
        rewards,
        labels,
        available_actions,
        observations,
        rates,
        valuations,
        modeltype,
    )

    # we create the model with the given type and initial state
    model = stormvogel.model.new_model(modeltype=modeltype, create_initial_state=False)
    init = model.new_state(labels=["init"])

    # we continue calling delta and adding new states until no states are
    # left to be checked
    states_seen = []
    states_to_be_visited = [initial_state_pgc]
    state_lookup = {initial_state_pgc: init}
    while len(states_to_be_visited) > 0:
        state = states_to_be_visited[0]
        states_to_be_visited.remove(state)
        transition = {}

        if state not in states_seen:
            states_seen.append(state)

        if model.supports_actions():
            # we loop over all available actions and call the delta function for each actions
            assert available_actions is not None
            actionslist = available_actions(state)

            if actionslist is None:
                raise ValueError(
                    f"On input {state}, the available actions function does not have a return value"
                )

            if not isinstance(actionslist, list):
                raise ValueError(
                    f"On input {state}, the available actions function does not return a list. Make sure to change it to [{actionslist}]"
                )

            for action in actionslist:
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

                if not isinstance(tuples, list) and tuples is not None:
                    raise ValueError(
                        f"On input pair {state} {action}, the delta function does not return a list. Make sure to change the format to [(<value>,<state>),...]"
                    )

                branch = add_new_transitions(tuples, states_seen, state)

                if branch != []:
                    transition[stormvogel_action] = stormvogel.model.Branch(branch)
        else:
            tuples = delta(state)

            if not isinstance(tuples, list) and tuples is not None:
                raise ValueError(
                    f"On input {state}, the delta function does not return a list. Make sure to change the format to [(<value>,<state>),...]"
                )

            branch = add_new_transitions(tuples, states_seen, state)

            if branch != []:
                transition[stormvogel.model.EmptyAction] = stormvogel.model.Branch(
                    branch
                )

        s = state_lookup[state]
        assert s is not None
        model.add_transitions(
            s,
            stormvogel.model.Transition(transition),
        )

        # if at some point we discovered more than max_size states, we complain
        if len(states_seen) > max_size:
            raise RuntimeError(
                f"The model you want te create has a very large amount of states (at least {max_size}), if you wish to proceed, set max_size to some larger number."
            )

    # we add the rewards
    if rewards is not None:
        if model.supports_actions():
            # we first create the right number of reward models
            assert available_actions is not None
            for reward in rewards(
                initial_state_pgc, available_actions(initial_state_pgc)[0]
            ).items():
                model.add_rewards(reward[0])

            action = available_actions(initial_state_pgc)[0]
            initial_state_rewards = rewards(initial_state_pgc, action)
            for state in states_seen:
                assert available_actions is not None
                for action in available_actions(state):
                    rewarddict = rewards(state, action)

                    # we check for the rewards when the function does not return a dict object
                    # or the length is not always the same
                    if rewarddict is None:
                        raise ValueError(
                            f"On input pair {state} {action}, the rewards function does not have a return value"
                        )

                    if not isinstance(rewarddict, dict):
                        raise ValueError(
                            f"On input pair {state} {action}, the rewards function does not return a dictionary. Make sure to change it to the format {{<rewardmodel>:<reward>,...}}"
                        )
                    if rewarddict.keys() != initial_state_rewards.keys():
                        raise ValueError(
                            "Make sure that the rewards function returns a dictionary with the same keys on each return"
                        )

                    s = state_lookup[state]
                    assert s is not None
                    for index, reward in enumerate(rewarddict.items()):
                        a = model.get_action_with_labels(frozenset(action.labels))
                        assert a is not None
                        model.rewards[index].set_state_action_reward(
                            s,
                            a,
                            reward[1],
                        )
        else:
            # we first create the right number of reward models
            for reward in rewards(initial_state_pgc).items():
                model.add_rewards(reward[0])

            initial_state_rewards = rewards(initial_state_pgc)
            for state in states_seen:
                rewarddict = rewards(state)

                # we check for the rewards when the function does not return a dict object
                # or the length is not always the same
                if rewarddict is None:
                    raise ValueError(
                        f"On input {state}, the rewards function does not have a return value"
                    )

                if not isinstance(rewarddict, dict):
                    raise ValueError(
                        f"On input {state}, the rewards function does not return a dictionary. Make sure to change it to the format {{<rewardmodel name>:<reward>,...}}"
                    )
                if rewarddict.keys() != initial_state_rewards.keys():
                    raise ValueError(
                        "Make sure that the rewards function returns a dictionary with the same keys on each return"
                    )

                s = state_lookup[state]
                assert s is not None
                for index, reward in enumerate(rewarddict.items()):
                    model.rewards[index].set_state_reward(s, reward[1])

    # we add the observations
    if observations is not None:
        for state in states_seen:
            s = state_lookup[state]

            # we check for the observations when it does not return an integer
            o = observations(state)
            if o is None:
                raise ValueError(
                    f"On input {state}, the observations function does not have a return value"
                )

            if not isinstance(o, int):
                raise ValueError(
                    f"On input {state}, the observations function does not return an integer"
                )

            s.set_observation(o)

    # we add the exit rates
    if rates is not None:
        for state in states_seen:
            s = state_lookup[state]

            r = rates(state)
            if not isinstance(r, stormvogel.model.Number):
                raise ValueError(
                    f"On input {state}, the rates function does not return a number"
                )
            model.set_rate(s, r)

    # we add the valuations
    if valuations is not None:
        initial_state_valuations = valuations(initial_state_pgc)
        for state in states_seen:
            s = state_lookup[state]

            valuation_list = valuations(state)
            if valuation_list is None:
                raise ValueError(
                    f"On input {state}, the valuations function does not have a return value"
                )

            if not isinstance(valuation_list, dict):
                raise ValueError(
                    f"On input {state}, the valuations function does not return a dictionary. Make sure to change the format to [<variable>: <value>,...]"
                )

            if valuation_list.keys() != initial_state_valuations.keys():
                raise RuntimeError(
                    "Make sure that you have a value for each variable in each state"
                )

            for val in valuation_list.values():
                if not (
                    isinstance(val, int)
                    or isinstance(val, bool)
                    or isinstance(val, float)
                ):
                    raise ValueError(
                        f"On input {state}, the dictionary that the valuations function returns contains a value {val} which is not of type int, float or"
                    )

            s.valuations = valuation_list

    # we add the labels
    if labels is not None:
        for state in states_seen:
            labellist = labels(state)

            # we check for the labels when the function does not return a list object
            # or the length is not always the same
            if labellist is None:
                raise ValueError(
                    f"On input {state}, the labels function does not have a return value"
                )

            s = state_lookup[state]
            assert s is not None
            if not isinstance(
                labellist, list
            ):  # if we don't get a list, we assume there is just one label
                s.add_label(labellist)
            else:
                for label in labellist:
                    s.add_label(label)

    return model
