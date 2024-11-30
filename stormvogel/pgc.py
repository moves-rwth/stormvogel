import stormvogel.model
from dataclasses import dataclass
from typing import Callable
import math


@dataclass
class State:
    x: int

    def __hash__(self):
        return hash(self.x)

    def __eq__(self, other):
        if isinstance(other, State):
            return self.x == other.x
        return False


@dataclass
class Action:
    labels: list[str]


def build_pgc(
    delta,  # Callable[[State, Action], list[tuple[float, State]]],
    initial_state_pgc: State,
    available_actions: Callable[[State], list[Action]],
    modeltype: stormvogel.model.ModelType = stormvogel.model.ModelType.MDP,
) -> stormvogel.model.Model:
    """
    function that converts a delta function, an available_actions function an initial state and a model type
    to a stormvogel model

    this works analogous to a prism file, where the delta is the module in this case.

    (this function uses the pgc classes State and Action instead of the ones from stormvogel.model)

    (currently only works for mdps)
    """

    model = stormvogel.model.new_model(modeltype=modeltype, create_initial_state=False)

    # we create the model with the given type and initial state
    model.new_state(
        labels=["init", str(initial_state_pgc.x)], features={"x": initial_state_pgc.x}
    )

    # we continue calling delta and adding new states until no states are
    # left to be checked
    states_discovered = []
    states_to_be_discovered = [initial_state_pgc]
    while len(states_to_be_discovered) > 0:
        state = states_to_be_discovered[0]
        states_to_be_discovered.remove(state)
        states_discovered.append(state)
        # we loop over all available actions and call the delta function for each action
        transition = {}
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
            # we add all the newly found transitions to the model (including the action)
            branch = []
            for tuple in tuples:
                if tuple[1] not in states_discovered:
                    new_state = model.new_state(
                        labels=[str(tuple[1].x)], features={"x": tuple[1].x}
                    )
                    branch.append((tuple[0], new_state))
                    states_to_be_discovered.append(tuple[1])
                else:
                    # TODO what if there are multiple states with the same label? use names?
                    branch.append(
                        (tuple[0], model.get_states_with_label(str(tuple[1].x))[0])
                    )
            if branch != []:
                transition[stormvogel_action] = stormvogel.model.Branch(branch)
        model.add_transitions(
            model.get_states_with_label(str(state.x))[0],
            stormvogel.model.Transition(transition),
        )

    return model


"""
def build(
    delta: Callable[
        [stormvogel.model.State, stormvogel.model.Action],
        list[tuple[float, stormvogel.model.State]],
    ],
    available_actions: Callable[
        [stormvogel.model.State], list[stormvogel.model.Action]
    ],
    initial_value: str,
    modeltype: stormvogel.model.ModelType | None = stormvogel.model.ModelType.MDP,
) -> stormvogel.model.Model:

    #function that converts a delta function, an available_actions function an initial state and a model type
    #to a stormvogel model

    #this works analogous to a prism file, where the delta is the module in this case.

    # we create the model with the given type and initial state
    model = stormvogel.model.new_model(modeltype, create_initial_state=False)
    model.new_state(features=[initial_state.features])

    # we continue calling delta and adding new states until no states are
    # left to be checked
    states = [initial_state]
    for state in states:
        # we loop over all available actions and call the delta function for each action
        transition = {}
        for action in available_actions(state):
            tuples = delta(state, action)
            # we add all the newly found transitions to the model (including the action)
            branch = []
            for tuple in tuples:
                branch.append(tuple)
                if tuple[1] not in states:
                    states.add(tuple[1])
            transition[action] = branch
        model.add_transitions(state, stormvogel.model.Transition(transition))

    return model
"""

if __name__ == "__main__":
    N = 100
    p = 0.5
    initial_state = State(math.floor(N / 2))

    left = Action(["left"])
    right = Action(["right"])

    def available_actions(s: State) -> list[Action]:
        return [left, right]

    def delta(s: State, action: Action):
        if action == left:
            return [(p, State(x=s.x + 1)), (1 - p, State(x=s.x))] if s.x < N else []
        elif action == right:
            return [(p, State(x=s.x - 1)), (1 - p, State(x=s.x))] if s.x > 0 else []

    model = build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
    )

    print(model)

    # Do we also want to be able to do it with the stormvogel.model objects?
    """
    N = 10
    p = 0.5

    initial_value = 5

    left = stormvogel.model.Action("",["left"])
    right = stormvogel.modle.Action("",["right"])

    def available_actions(s: stormvogel.model.State) -> list[stormvogel.model.Action]:
        return [left, right]

    def delta(s: stormvogel.model.State, action: stormvogel.model.Action) -> stormvogel.model.State:
        if action == left:
            return [(p, stormvogel.model.State(features={x:s.x + 1})), (1 - p, stormvogel.model.State(features=["s.x"]))]
        elif action == right:
            return [(p, stormvogel.model.State(features=["s.x - 1"])), (1 - p, stormvogel.model.State(features=["s.x"]))]



    model = build(delta, available_actions, initial_value, modeltype)

    print(model)
    """
