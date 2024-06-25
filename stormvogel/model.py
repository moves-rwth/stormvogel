"""Contains the python model representations and their APIs."""

from dataclasses import dataclass
from enum import Enum
from numbers import Number


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

    def set_transitions(self, transitions: "Transition" | "TransitionShorthand"):
        """Set transitions from this state."""
        self.model.set_transitions(self, transitions)

    def add_transitions(self, transitions: "Transition" | "TransitionShorthand"):
        """Add transitions from this state."""
        self.model.add_transitions(self, transitions)


@dataclass
class Action:
    """Represents an action, e.g., in MDPs.

    Args:
        name: A name for this action.
        labels: The labels of this action. Corresponds to Storm labels.
    """

    name: str
    labels: list[str]


# The empty action. Used for DTMCs.
EmptyAction = Action(None, [])


@dataclass
class Branch:
    """Represents a branch, which is a distribution over states.

    Args:
        branch: The branch.
    """

    branch: list[tuple[Number, State]]


@dataclass
class Transition:
    """Represents a transition, which map actions to branches.

    Args:
        transition: The transition.
    """

    transition: dict[Action, Branch]


# How the user is allowed to specify a transition:
# - using only the action and the target state (implies probability=1),
# - using only the probability and the target state (implies default action when in an MDP),
type TransitionShorthand = list[tuple[Action, State]] | list[tuple[Number, State]]


def transition_from_shorthand(shorthand: TransitionShorthand) -> Transition:
    """Get a Transition object from a TransitionShorthand."""
    if isinstance(shorthand, list[tuple[Action, State]]):
        return Transition(
            {action: Branch([(1, state)]) for (action, state) in shorthand}
        )
    elif isinstance(shorthand, list[tuple[Number, State]]):
        return Transition({EmptyAction: Branch(shorthand)})


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
    states: dict[int, State] = {}
    actions: dict[str, State] | None = None
    transitions: dict[State, Transition] = {}

    def __init__(self, name: str | None, model_type: ModelType):
        self.name = name
        self.type = model_type
        # Add the initial state
        self.states = {0: State("init", ["init"], {}, 0, self)}
        # Create a self-loop at the initial state
        self.get_state_by_id(0).set_transitions([(1, self.get_state_by_id(0))])
        # Initialize actions if those are supported by the model type
        if self.__supports_actions():
            self.actions = {}

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
        if isinstance(transitions, TransitionShorthand):
            transitions = transition_from_shorthand(transitions)
        self.transitions[s] = transitions

    def add_transitions(self, s: State, transitions: Transition | TransitionShorthand):
        """Add new transitions from a state."""
        assert self.type in [ModelType.MDP, ModelType.MA]
        if isinstance(transitions, TransitionShorthand):
            transitions = transition_from_shorthand(transitions)
        for choice, branch in transitions.choices.items():
            self.transitions[s].choices[choice] = branch

    def new_action(self, name: str, labels: list[str] | None = None) -> Action:
        """Creates a new action and returns it."""
        if not self.__supports_actions():
            raise RuntimeError(
                "Called new_action on a model that does not support actions"
            )
        assert self.actions
        if name in self.actions:
            raise RuntimeError(
                f"Tried to add action {name} but that action already exists"
            )
        action = Action(name, labels or [])
        self.actions[name] = action

    def get_action(self, name: str) -> Action:
        """Gets an existing action."""
        if not self.__supports_actions():
            raise RuntimeError(
                "Called get_action on a model that does not support actions"
            )
        assert self.actions
        if not name in self.actions:
            raise RuntimeError(
                f"Tried to get action {name} but that action does not exist"
            )
        return self.actions[name]

    def new_state(
        self,
        name: str | None = None,
        labels: list[str] | None = None,
        features: dict[str, any] | None = None,
    ) -> State:
        """Creates a new state and returns it."""
        state_id = self.__free_state_id()
        state = State(name, labels or [], features or set(), state_id, self)
        self.states[state_id] = state
        return state

    def get_states_with(self, label: str):
        """Get all states with a given label."""
        # TODO: slow, not sure if that will become a problem though
        collected_states = set()
        for _id, state in self.states.items():
            if label in state.labels:
                collected_states.add(state)
        return collected_states

    def get_state_by_id(self, state_id):
        """Get a state by its id."""
        if not state_id in self.states:
            raise RuntimeError("Requested a non-existing state")
        return self.states[state_id]

    def get_initial_state(self):
        """Gets the initial state (id=0)."""
        return self.states[0]

    def get_labels(self):
        """Get all labels in states of this Model."""
        collected_labels = set()
        for _id, state in self.states.items():
            collected_labels += state.labels
        return collected_labels


def new_dtmc(name: str | None = None):
    """Creates a DTMC."""
    return Model(name, ModelType.MDP)


def new_mdp(name: str | None = None):
    """Creates an MDP."""
    return Model(name, ModelType.MDP)
