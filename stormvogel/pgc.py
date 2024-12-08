import stormvogel.model
from dataclasses import dataclass
from typing import Callable


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


def build_pgc(
    delta,  # Callable[[State, Action], list[tuple[float, State]]],
    initial_state_pgc: State,  # TODO rewards function, label function
    available_actions: Callable[[State], list[Action]] | None = None,
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
) -> stormvogel.model.Model:
    """
    function that converts a delta function, an available_actions function an initial state and a model type
    to a stormvogel model

    this works analogous to a prism file, where the delta is the module in this case.

    (this function uses the pgc classes State and Action instead of the ones from stormvogel.model)

    (currently only works for mdps)
    """
    if modeltype == stormvogel.model.ModelType.MDP and available_actions is None:
        raise RuntimeError(
            "You have to provide an available actions function for mdp models"
        )

    model = stormvogel.model.new_model(modeltype=modeltype, create_initial_state=False)

    # we create the model with the given type and initial state
    model.new_state(
        labels=["init"],
        features=initial_state_pgc.__dict__,
        name=str(initial_state_pgc.__dict__),
    )

    # we continue calling delta and adding new states until no states are
    # left to be checked
    states_seen = []
    states_to_be_visited = [initial_state_pgc]
    while len(states_to_be_visited) > 0:
        state = states_to_be_visited[0]
        states_to_be_visited.remove(state)
        # we loop over all available actions and call the delta function for each action
        transition = {}

        if state not in states_seen:
            states_seen.append(state)

        if model.supports_actions():
            assert available_actions is not None
            for action in available_actions(state):
                try:
                    stormvogel_action = model.new_action(
                        str(action.labels),
                        frozenset(
                            {action.labels[0]}
                        ),  # right now we only look at one label
                    )
                except RuntimeError:
                    stormvogel_action = model.get_action(str(action.labels))

                tuples = delta(state, action)
                # we add all the newly found transitions to the model
                branch = []
                for tuple in tuples:
                    if tuple[1] not in states_seen:
                        states_seen.append(tuple[1])
                        new_state = model.new_state(
                            name=str(tuple[1].__dict__), features=tuple[1].__dict__
                        )
                        branch.append((tuple[0], new_state))
                        states_to_be_visited.append(tuple[1])
                    else:
                        # print(tuple[1].__dict__)
                        # print(model.states)
                        branch.append(
                            (tuple[0], model.get_state_by_name(str(tuple[1].__dict__)))
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
                    new_state = model.new_state(
                        name=str(tuple[1].__dict__), features=tuple[1].__dict__
                    )

                    branch.append((tuple[0], new_state))
                    states_to_be_visited.append(tuple[1])
                else:
                    branch.append(
                        (tuple[0], model.get_state_by_name(str(tuple[1].__dict__)))
                    )
                if branch != []:
                    transition[stormvogel.model.EmptyAction] = stormvogel.model.Branch(
                        branch
                    )
        s = model.get_state_by_name(str(state.__dict__))
        assert s is not None
        model.add_transitions(
            s,
            stormvogel.model.Transition(transition),
        )

    return model
