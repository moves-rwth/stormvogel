"""Contains the python model representations and their APIs."""

from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """The type of the model."""

    # implemented
    DTMC = 1
    MDP = 2
    # not implemented yet
    CTMC = 3
    MA = 4


@dataclass
class State:
    """Represents a state in a Model.

    Args:
        name: An optional name for this state.
        labels: The labels of this state. Corresponds to Storm labels.
        features: The features of this state. Corresponds to Storm features.
        id: The number of this state in the matrix.
        model: The model this state belongs to.
    """

    labels: list[str]
    features: dict[str, int]
    id: int
    model: "Model"

    def set_transitions(self, transitions: "Transition | TransitionShorthand"):
        """Set transitions from this state."""
        self.model.set_transitions(self, transitions)

    def add_transitions(self, transitions: "Transition | TransitionShorthand"):
        """Add transitions from this state."""
        self.model.add_transitions(self, transitions)

    def __str__(self):
        return f"State {self.id} with labels {self.labels} and features {self.features}"


@dataclass(frozen=True)
class Action:
    """Represents an action, e.g., in MDPs.

    Args:
        name: A name for this action.
        labels: The labels of this action. Corresponds to Storm labels.
    """

    name: str

    def __str__(self):
        return f"Action {self.name}"


# The empty action. Used for DTMCs.
EmptyAction = Action("empty")


@dataclass
class Branch:
    """Represents a branch, which is a distribution over states.

    Args:
        branch: The branch.
    """

    branch: list[tuple[float, State]]

    def __str__(self):
        parts = []
        for prob, state in self.branch:
            parts.append(f"{prob} -> {state}")
        return ", ".join(parts)


@dataclass
class Transition:
    """Represents a transition, which map actions to branches.

    Args:
        transition: The transition.
    """

    transition: dict[Action, Branch]

    def __str__(self):
        parts = []
        for action, branch in self.transition.items():
            if action == EmptyAction:
                parts.append(f"{branch}")
            else:
                parts.append(f"{action} => {branch}")
        return "; ".join(parts + [])


# How the user is allowed to specify a transition:
# - using only the action and the target state (implies probability=1),
# - using only the probability and the target state (implies default action when in an MDP),
type TransitionShorthand = list[tuple[float, State]] | list[tuple[float, State]]


def transition_from_shorthand(shorthand: TransitionShorthand) -> Transition:
    """Get a Transition object from a TransitionShorthand."""
    if len(shorthand) == 0:
        raise RuntimeError("Transition cannot be empty")
    # Check the type of the first element
    first_element = shorthand[0][0]
    if isinstance(first_element, Action):
        transition_content = {}
        for action, state in shorthand:
            assert isinstance(action, Action)
            transition_content[action] = Branch([(1, state)])
        return Transition(transition_content)
    elif isinstance(first_element, float) or isinstance(first_element, int):
        return Transition({EmptyAction: Branch(shorthand)})
    raise RuntimeError(
        f"Type of {first_element} not supported in transition {shorthand}"
    )


@dataclass
class Model:
    """Represents a model.

    Args:
        name: An optional name for this model.
        type: The model type.
        states: The states of the model. The keys are the state's ids.
        actions: The actions of the model, if this is a model that supports actions.
        transitions: The transitions of this model.
    """

    name: str | None
    type: ModelType
    # Both of these are hashed by the id of the state (=number in the matrix)
    states: dict[int, State]
    transitions: dict[int, Transition]
    actions: dict[str, Action] | None

    def __init__(self, name: str | None, model_type: ModelType):
        self.name = name
        self.type = model_type
        self.transitions = {}
        self.states = {}

        # Initialize actions if those are supported by the model type
        if self.__supports_actions():
            self.actions = {}

        # Add the initial state
        self.new_state(["init"])

    def __supports_actions(self):
        """Returns whether this model supports actions."""
        return self.type in (ModelType.MDP, ModelType.MA)

    def __free_state_id(self):
        """Gets a free id in the states dict."""
        # TODO: slow, not sure if that will become a problem though
        i = 0
        while i in self.states:
            i += 1
        return i

    def set_transitions(self, s: State, transitions: Transition | TransitionShorthand):
        """Set the transition from a state."""
        if not isinstance(transitions, Transition):
            transitions = transition_from_shorthand(transitions)
        self.transitions[s.id] = transitions

    def add_transitions(self, s: State, transitions: Transition | TransitionShorthand):
        """Add new transitions from a state."""
        if not self.__supports_actions():
            raise RuntimeError(
                "In a model that does not support actions, you have to set transitions, not add them"
            )
        if not isinstance(transitions, Transition):
            transitions = transition_from_shorthand(transitions)
        for choice, branch in transitions.transition.items():
            self.transitions[s.id].transition[choice] = branch

    def new_action(self, name: str) -> Action:
        """Creates a new action and returns it."""
        if not self.__supports_actions():
            raise RuntimeError(
                "Called new_action on a model that does not support actions"
            )
        assert self.actions is not None
        if name in self.actions:
            raise RuntimeError(
                f"Tried to add action {name} but that action already exists"
            )
        action = Action(name)
        self.actions[name] = action
        return action

    def get_action(self, name: str) -> Action:
        """Gets an existing action."""
        if not self.__supports_actions():
            raise RuntimeError(
                "Called get_action on a model that does not support actions"
            )
        assert self.actions is not None
        if name not in self.actions:
            raise RuntimeError(
                f"Tried to get action {name} but that action does not exist"
            )
        return self.actions[name]

    def action(self, name: str) -> Action:
        """New action or get action if it exists."""
        if not self.__supports_actions():
            raise RuntimeError(
                "Called get_action on a model that does not support actions"
            )
        assert self.actions is not None
        if name in self.actions:
            return self.get_action(name)
        else:
            return self.new_action(name)

    def new_state(
        self,
        labels: list[str] | str | None = None,
        features: dict[str, int] | None = None,
    ) -> State:
        """Creates a new state and returns it."""
        state_id = self.__free_state_id()
        if isinstance(labels, list):
            state = State(labels, features or {}, state_id, self)
        elif isinstance(labels, str):
            state = State([labels], features or {}, state_id, self)
        elif labels is None:
            state = State([], features or {}, state_id, self)

        self.states[state_id] = state

        # Create a self-loop at this state
        state.set_transitions([(1, state)])

        return state

    def get_states_with(self, label: str):
        """Get all states with a given label."""
        # TODO: slow, not sure if that will become a problem though
        collected_states = []
        for _id, state in self.states.items():
            if label in state.labels:
                collected_states.append(state)
        return collected_states

    def get_state_by_id(self, state_id):
        """Get a state by its id."""
        if state_id not in self.states:
            raise RuntimeError("Requested a non-existing state")
        return self.states[state_id]

    def get_initial_state(self):
        """Gets the initial state (id=0)."""
        return self.states[0]

    def get_labels(self):
        """Get all labels in states of this Model."""
        collected_labels: set[str] = set()
        for _id, state in self.states.items():
            collected_labels = collected_labels | set(state.labels)
        return collected_labels

    def to_dot(self) -> str:
        """Generates a dot representation of this model."""
        dot = "digraph model {\n"
        for state_id, state in self.states.items():
            dot += f'{state_id} [ label = "{state_id}: {", ".join(state.labels)}" ];\n'
        for state_id, transition in self.transitions.items():
            for action, branch in transition.transition.items():
                if action != EmptyAction:
                    dot += f'{action.name.replace(" ", "_")}{state_id} [ label = "", shape=point ];\n'
        for state_id, transition in self.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        dot += f'{state_id} -> {target.id} [ label = "{prob}" ];\n'
                else:
                    # Draw actions, then probabilities
                    dot += f'{state_id} -> {action.name.replace(" ", "_")}{state_id} [ label = "{action.name}" ];\n'
                    for prob, target in branch.branch:
                        dot += f'{action.name.replace(" ", "_")}{state_id} -> {target.id} [ label = "{prob}" ];\n'

        dot += "}"
        return dot

    def __str__(self) -> str:
        res = [f"{self.type} with name {self.name}"]
        res += ["", "States:"] + [f"{state}" for (_id, state) in self.states.items()]
        res += ["", "Transitions:"] + [
            f"{transition}" for (_id, transition) in self.transitions.items()
        ]

        return "\n".join(res)


def new_dtmc(name: str | None = None):
    """Creates a DTMC."""
    return Model(name, ModelType.MDP)


def new_mdp(name: str | None = None):
    """Creates an MDP."""
    return Model(name, ModelType.MDP)
