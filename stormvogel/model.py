"""Contains the python model representations and their APIs."""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import cast

Parameter = str

Number = float | Parameter | Fraction | int


class ModelType(Enum):
    """The type of the model."""

    # implemented
    DTMC = 1
    MDP = 2
    CTMC = 3
    POMDP = 4
    MA = 5


@dataclass
class Observation:
    """Represents an observation of a state (for pomdps)

    Args:
        observation: the observation as an integer
    """

    observation: int

    def get_observation(self) -> int:
        """returns the observation"""
        return self.observation

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.observation == other.observation
        return False

    def __str__(self):
        return f"Observation: {self.observation}"


@dataclass()
class State:
    """Represents a state in a Model.

    Args:
        labels: The labels of this state. Corresponds to Storm labels.
        features: The features of this state. Corresponds to Storm features.
        id: The number of this state in the matrix.
        model: The model this state belongs to.
        observation: the observation of this state in case the model is a pomdp.
        name: the name of this state.
    """

    labels: list[str]
    features: dict[str, int]
    id: int
    model: "Model"
    observation: Observation | None
    name: str

    def __init__(
        self,
        labels: list[str],
        features: dict[str, int],
        id: int,
        model,
        name: str | None = None,
    ):
        self.model = model

        if id in self.model.states.keys():
            raise RuntimeError(
                "There is already a state with this id. Make sure the id is unique."
            )

        used_names = [state.name for state in self.model.states.values()]
        if name in used_names:
            raise RuntimeError(
                "There is already a state with this name. Make sure the name is unique."
            )

        self.labels = labels
        self.features = features
        self.id = id
        self.observation = None

        if name is None:
            self.name = str(id)
        else:
            self.name = name

    def add_label(self, label: str):
        """adds a new label to the state"""
        if label not in self.labels:
            self.labels.append(label)

    def set_observation(self, observation: int) -> Observation:
        """sets the observation for this state"""
        if self.model.get_type() == ModelType.POMDP:
            self.observation = Observation(observation)
            return self.observation
        else:
            raise RuntimeError("The model this state belongs to is not a pomdp")

    def get_observation(self) -> Observation:
        """gets the observation"""
        if self.model.supports_observations():
            if self.observation is not None:
                return self.observation
            else:
                raise RuntimeError(
                    "This state does not have an observation yet. Add one with the new_observation function."
                )
        else:
            raise RuntimeError(
                "The model this state belongs to does not support observations"
            )

    def set_transitions(self, transitions: "Transition | TransitionShorthand"):
        """Set transitions from this state."""
        self.model.set_transitions(self, transitions)

    def add_transitions(self, transitions: "Transition | TransitionShorthand"):
        """Add transitions from this state."""
        self.model.add_transitions(self, transitions)

    def available_actions(self) -> list["Action"]:
        """returns the list of all available actions in this state"""
        if self.model.supports_actions():
            action_list = []
            for action in self.model.transitions[self.id].transition.keys():
                action_list.append(action)
            return action_list
        else:
            return [EmptyAction]

    def get_outgoing_transitions(
        self, action: "Action | None" = None
    ) -> list[tuple[Number, "State"]] | None:
        """gets the outgoing transitions"""
        if action and self.model.supports_actions():
            if self.id in self.model.transitions.keys():
                branch = self.model.transitions[self.id].transition[action]
                return branch.branch
        elif self.model.supports_actions() and not action:
            raise RuntimeError("You need to provide a specific action")
        else:
            if self.id in self.model.transitions.keys():
                branch = self.model.transitions[self.id].transition[EmptyAction]
                return branch.branch
        return None

    def is_absorbing(self, action: "Action | None" = None) -> bool:
        """returns if the state has a nonzero transition going to another state or not"""
        transitions = self.get_outgoing_transitions(action)
        if transitions is not None:
            for transition in transitions:
                if float(transition[0]) > 0 and transition[1] != self:
                    return False
        return True

    def __str__(self):
        res = f"State {self.id} with labels {self.labels} and features {self.features}"
        if self.model.supports_observations() and self.observation is not None:
            res += f" and observation {self.observation.get_observation()}"
        return res

    def __eq__(self, other):
        if isinstance(other, State):
            if self.id == other.id:
                if self.model.supports_observations():
                    if self.observation is not None and other.observation is not None:
                        observations_equal = self.observation == other.observation
                    else:
                        observations_equal = True
                else:
                    observations_equal = True
                return (
                    sorted(self.labels) == sorted(other.labels) and observations_equal
                )
            return False
        return False

    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return str(self.id) < str(other.id)


@dataclass(frozen=True, order=True)
class Action:
    """Represents an action, e.g., in MDPs.
        Note that this action object is completely independent of its corresponding branch.
        Their relation is managed by Transitions.

    Args:
        name: A name for this action.
        labels: The labels of this action. Corresponds to Storm labels.
    """

    name: str
    labels: frozenset[str]

    def __str__(self):
        return f"Action {self.name} with labels {self.labels}"


# The empty action. Used for DTMCs and empty action transitions in mdps.
EmptyAction = Action("empty", frozenset())


@dataclass(order=True)
class Branch:
    """Represents a branch, which is a distribution over states.

    Args:
        branch: The branch as a list of tuples.
            The first element is the probability and the second element is the target state.
    """

    branch: list[tuple[Number, State]]

    def __str__(self):
        parts = []
        for prob, state in self.branch:
            parts.append(f"{prob} -> {state}")
        return ", ".join(parts)

    def __eq__(self, other):
        if isinstance(other, Branch):
            return sorted(self.branch) == sorted(other.branch)
        return False

    def __add__(self, other):
        return Branch(self.branch + other.branch)


class Transition:
    """Represents a transition, which map actions to branches.
        Note that an EmptyAction may be used if we want a non-action transition.
        Note that a single Transition might correspond to multiple 'arrows'.

    Args:
        transition: The transition.
    """

    transition: dict[Action, Branch]

    def __init__(self, transition: dict[Action, Branch]):
        # Input validation, see RuntimeError.
        if len(transition) > 1 and EmptyAction in transition:
            raise RuntimeError(
                "It is impossible to create a transition that contains more than one action, and an emtpy action"
            )
        self.transition = transition

    def __str__(self):
        parts = []
        for action, branch in self.transition.items():
            if action == EmptyAction:
                parts.append(f"{branch}")
            else:
                parts.append(f"{action} => {branch}")
        return "; ".join(parts + [])

    def __eq__(self, other):
        if isinstance(other, Transition):
            return sorted(list(self.transition.values())) == sorted(
                list(other.transition.values())
            )
        return False

    def has_empty_action(self) -> bool:
        # Note that we don't have to deal with the corner case where there are both empty and non-empty transitions. This is dealt with at __init__.
        return self.transition.keys() == {EmptyAction}


TransitionShorthand = list[tuple[Number, State]] | list[tuple[Action, State]]


def transition_from_shorthand(shorthand: TransitionShorthand) -> Transition:
    """Get a Transition object from a TransitionShorthand. Use for all transitions in DTMCs and for empty actions in MDPs.

    There are two possible ways to define a TransitionShorthand.
    - using only the probability and the target state (implies default action when in an MDP).
    - using only the action and the target state (implies probability=1)."""
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
    elif (
        isinstance(first_element, float)
        or isinstance(first_element, int)
        or isinstance(first_element, Fraction)
        or isinstance(first_element, str)
    ):
        return Transition(
            {EmptyAction: Branch(cast(list[tuple[Number, State]], shorthand))}
        )
    raise RuntimeError(
        f"Type of {first_element} not supported in transition {shorthand}"
    )


@dataclass(order=True)
class RewardModel:
    """Represents a state-exit reward model.
    Args:
        name: Name of the reward model.
        rewards: The rewards, the keys are the state's ids (or state action pair ids).
    """

    name: str
    # Hashed by the id of the state or state action pair (=number in the matrix)
    rewards: dict[int, Number]

    def get(self, state: State) -> Number:
        """Gets the reward at said state."""
        return self.rewards[state.id]

    def set(self, state: State, value: Number):
        """Sets the reward at said state."""
        self.rewards[state.id] = value

    def set_action_state(self, state_action_pair: int, value: Number):
        """sets the reward at said state action pair"""
        self.rewards[state_action_pair] = value


@dataclass
class Model:
    """Represents a model.

    Args:
        name: An optional name for this model.
        type: The model type.
        states: The states of the model. The keys are the state's ids.
        transitions: The transitions of this model.
        actions: The actions of the model, if this is a model that supports actions.
        rewards: The rewardsmodels of this model.
        exit_rates: The exit rates of the model, optional if this model supports rates.
        markovian_states: list of markovian states in the case of a ma.
    """

    name: str | None
    type: ModelType
    # Both of these are hashed by the id of the state (=number in the matrix)
    states: dict[int, State]
    transitions: dict[int, Transition]
    actions: dict[str, Action] | None
    rewards: list[RewardModel]
    # In ctmcs we work with rate transitions but additionally we can optionally store exit rates (hashed by id of the state)
    exit_rates: dict[int, Number] | None
    # In ma's we keep track of markovian states
    markovian_states: list[State] | None

    def __init__(
        self, name: str | None, model_type: ModelType, create_initial_state: bool = True
    ):
        self.name = name
        self.type = model_type
        self.transitions = {}
        self.states = {}
        self.rewards = []

        # Initialize actions if those are supported by the model type
        if self.supports_actions():
            self.actions = {}
        else:
            self.actions = None

        # Initialize rates if those are supported by the model type
        if self.supports_rates():
            self.exit_rates = {}
        else:
            self.exit_rates = None

        # Initialize observations if those are supported by the model type (pomdps)
        if self.get_type() == ModelType.POMDP:
            self.observations = {}
        else:
            self.observations = None

        # Initialize markovian states if applicable (in the case of MA's)
        if self.get_type() == ModelType.MA:
            self.markovian_states = []
        else:
            self.markovian_states = None

        # Add the initial state if specified to do so
        if create_initial_state:
            self.new_state(["init"])

    def supports_actions(self):
        """Returns whether this model supports actions."""
        return self.type in (ModelType.MDP, ModelType.POMDP, ModelType.MA)

    def supports_rates(self):
        """Returns whether this model supports rates."""
        return self.type in (ModelType.CTMC, ModelType.MA)

    def supports_observations(self):
        """Returns whether this model supports observations."""
        return self.type == ModelType.POMDP

    def is_stochastic(self) -> bool:
        """For discrete models: Checks if all sums of outgoing transition probabilities for all states equal 1
        For continuous models: Checks if all sums of outgoing rates sum to 0
        """

        if not self.supports_rates():
            for state in self.states.values():
                for action in state.available_actions():
                    sum_prob = 0
                    transitions = state.get_outgoing_transitions(action)
                    assert transitions is not None
                    for transition in transitions:
                        if (
                            isinstance(transition[0], float)
                            or isinstance(transition[0], Fraction)
                            or isinstance(transition[0], int)
                        ):
                            sum_prob += transition[0]
                    if sum_prob != 1:
                        return False
        else:
            for state in self.states.values():
                for action in state.available_actions():
                    sum_rates = 0
                    transitions = state.get_outgoing_transitions(action)
                    assert transitions is not None
                    for transition in transitions:
                        if (
                            isinstance(transition[0], float)
                            or isinstance(transition[0], Fraction)
                            or isinstance(transition[0], int)
                        ):
                            sum_rates += transition[0]
                    if sum_rates != 0:
                        return False

        return True

    def normalize(self):
        """Normalizes a model (for states where outgoing transition probabilities don't sum to 1, we divide each probability by the sum)"""
        if not self.supports_rates():
            self.add_self_loops()
            for state in self.states.values():
                for action in state.available_actions():
                    sum_prob = 0
                    transitions = state.get_outgoing_transitions(action)
                    assert transitions is not None
                    for tuple in transitions:
                        if (
                            isinstance(tuple[0], float)
                            or isinstance(tuple[0], Fraction)
                            or isinstance(tuple[0], int)
                        ):
                            sum_prob += tuple[0]

                    new_transitions = []
                    for tuple in transitions:
                        if (
                            isinstance(tuple[0], float)
                            or isinstance(tuple[0], Fraction)
                            or isinstance(tuple[0], int)
                        ):
                            normalized_transition = (
                                tuple[0] / sum_prob,
                                tuple[1],
                            )
                            new_transitions.append(normalized_transition)
                    self.transitions[state.id].transition[
                        action
                    ].branch = new_transitions
        else:
            # for ctmcs and mas we currently only add self loops
            self.add_self_loops()

    def __free_state_id(self):
        """Gets a free id in the states dict."""
        # TODO: slow, not sure if that will become a problem though
        i = 0
        while i in self.states:
            i += 1
        return i

    def add_self_loops(self):
        """adds self loops to all states that do not have an outgoing transition"""
        for id, state in self.states.items():
            if self.transitions.get(id) is None:
                self.set_transitions(
                    state, [(float(0) if self.supports_rates() else float(1), state)]
                )

    def all_states_outgoing_transition(self) -> bool:
        """checks if all states have an outgoing transition"""
        for state in self.states.items():
            if self.transitions.get(state[0]) is None:
                return False
        return True

    def add_markovian_state(self, markovian_state: State):
        """Adds a state to the markovian states."""
        if self.get_type() == ModelType.MA and self.markovian_states is not None:
            self.markovian_states.append(markovian_state)
        else:
            raise RuntimeError("This model is not a MA")

    def set_transitions(self, s: State, transitions: Transition | TransitionShorthand):
        """Set the transition from a state."""
        if not isinstance(transitions, Transition):
            transitions = transition_from_shorthand(transitions)
        self.transitions[s.id] = transitions

    def add_transitions(self, s: State, transitions: Transition | TransitionShorthand):
        """Add new transitions from a state. If no transition currently exists, the result will be the same as set_transitions."""

        if not isinstance(transitions, Transition):
            transitions = transition_from_shorthand(transitions)

        try:
            existing_transitions = self.get_transitions(s)
        except KeyError:
            # Empty transitions case, act like set_transitions.
            self.set_transitions(s, transitions)
            return

        if not self.supports_actions():
            self.transitions[s.id].transition[
                EmptyAction
            ].branch += transitions.transition[EmptyAction].branch
        else:
            # Adding a transition is only valid if they are both empty or both non-empty.
            if (
                not transitions.has_empty_action()
                and existing_transitions.has_empty_action()
            ):
                raise RuntimeError(
                    "You cannot add a transition with an non-empty action to a transition which has an empty action. Use set_transition instead."
                )
            if (
                transitions.has_empty_action()
                and not existing_transitions.has_empty_action()
            ):
                raise RuntimeError(
                    "You cannot add a transition with an empty action to a transition which has no empty action. Use set_transition instead."
                )

            # Empty action case, add the branches together.
            if transitions.has_empty_action():
                self.transitions[s.id].transition[EmptyAction] += (
                    transitions.transition[EmptyAction]
                )
            else:
                for choice, branch in transitions.transition.items():
                    self.transitions[s.id].transition[choice] = branch

    def get_transitions(self, state_or_id: State | int) -> Transition:
        """Get the transition at state s. Throws a KeyError if not present."""
        if isinstance(state_or_id, State):
            return self.transitions[state_or_id.id]
        else:
            return self.transitions[state_or_id]

    def get_branch(self, state_or_id: State | int) -> Branch:
        """Get the branch at state s. Only intended for emtpy transitions, otherwise a RuntimeError is thrown."""
        s_id = state_or_id if isinstance(state_or_id, int) else state_or_id.id
        transition = self.transitions[s_id].transition
        if EmptyAction not in transition:
            raise RuntimeError("Called get_branch on a non-empty transition.")
        return transition[EmptyAction]

    def new_action(self, name: str, labels: frozenset[str] | None = None) -> Action:
        """Creates a new action and returns it."""
        if not self.supports_actions():
            raise RuntimeError(
                "Called new_action on a model that does not support actions"
            )
        assert self.actions is not None
        if name in self.actions:
            raise RuntimeError(
                f"Tried to add action {name} but that action already exists"
            )
        if labels:
            action = Action(name, labels)
        else:
            action = Action(name, frozenset())
        self.actions[name] = action
        return action

    def reassign_ids(self):
        """reassigns the ids of states, transitions and rates to be in order again"""
        self.states = {
            new_id: value
            for new_id, (old_id, value) in enumerate(sorted(self.states.items()))
        }

        self.transitions = {
            new_id: value
            for new_id, (old_id, value) in enumerate(sorted(self.transitions.items()))
        }

        if self.supports_rates and self.exit_rates is not None:
            self.exit_rates = {
                new_id: value
                for new_id, (old_id, value) in enumerate(
                    sorted(self.exit_rates.items())
                )
            }

    def remove_state(
        self, state: State, normalize: bool = True, reassign_ids: bool = True
    ):
        """properly removes a state, it can optionally normalize the model and reassign ids automatically"""
        if state in self.states.values():
            # we remove the state from the transitions
            # first we remove transitions that go into the state
            remove_actions_index = []
            for index, transition in self.transitions.items():
                for action, branch in transition.transition.items():
                    for index_tuple, tuple in enumerate(branch.branch):
                        # remove the tuple if it goes to the state
                        if tuple[1].id == state.id:
                            self.transitions[index].transition[action].branch.pop(
                                index_tuple
                            )

                    # if we have empty actions we need to remove those as well (later)
                    if branch.branch == []:
                        remove_actions_index.append((action, index))
            # here we remove those empty actions (this needs to happen after the other for loops)
            for action, index in remove_actions_index:
                self.transitions[index].transition.pop(action)
                # if we have no actions at all anymore, delete the transition
                if self.transitions[index].transition == {} and not index == state.id:
                    self.transitions.pop(index)

            # we remove transitions that come out of the state
            self.transitions.pop(state.id)

            # We remove the state
            self.states.pop(state.id)

            # we remove the exit rates from the state when applicable
            if self.supports_rates and self.exit_rates is not None:
                self.exit_rates.pop(state.id)

            # we remove the state from the markovian state list when applicable
            if self.get_type() == ModelType.MA and self.markovian_states is not None:
                if state in self.markovian_states:
                    self.markovian_states.remove(state)

            # we normalize the model if specified to do so
            if normalize:
                self.normalize()

            # we reassign the ids if specified to do so
            if reassign_ids:
                self.reassign_ids()
                for other_state in self.states.values():
                    if other_state.id > state.id:
                        other_state.id -= 1

    def remove_transitions_between_states(
        self, state0: State, state1: State, normalize: bool = True
    ):
        """
        Remove the transition(s) that start in state0 and go to state1.
        Only works on models that don't support actions.
        """
        if not self.supports_actions():
            for tuple in self.transitions[state0.id].transition[EmptyAction].branch:
                if tuple[1] == state1:
                    self.transitions[state0.id].transition[EmptyAction].branch.remove(
                        tuple
                    )
            # if we have empty objects we need to remove those as well
            if self.transitions[state0.id].transition[EmptyAction].branch == []:
                self.transitions.pop(state0.id)

            if normalize:
                self.normalize()
        else:
            raise RuntimeError(
                "This method only works for models that don't support actions."
            )

    def get_action(self, name: str) -> Action:
        """Gets an existing action."""
        if not self.supports_actions():
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
        if not self.supports_actions():
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

        return state

    def get_states_with_label(self, label: str) -> list[State]:
        """Get all states with a given label."""
        # TODO: slow, not sure if that will become a problem though
        collected_states = []
        for _id, state in self.states.items():
            if label in state.labels:
                collected_states.append(state)
        return collected_states

    def get_state_by_id(self, state_id) -> State:
        """Get a state by its id."""
        if state_id not in self.states:
            raise RuntimeError("Requested a non-existing state")
        return self.states[state_id]

    def get_state_by_name(self, state_name) -> State | None:
        """Get a state by its name."""
        names = [state.name for state in self.states.values()]
        if state_name not in names:
            raise RuntimeError("Requested a non-existing state")

        for state in self.states.values():
            if state.name == state_name:
                return state

    def get_initial_state(self) -> State:
        """Gets the initial state (id=0)."""
        return self.states[0]

    def get_labels(self) -> set[str]:
        """Get all labels in states of this Model."""
        collected_labels: set[str] = set()
        for _id, state in self.states.items():
            collected_labels = collected_labels | set(state.labels)
        return collected_labels

    def get_default_rewards(self) -> RewardModel:
        """Gets the default reward model, throws a RuntimeError if there is none."""
        if len(self.rewards) == 0:
            raise RuntimeError("This model has no reward models.")
        return self.rewards[0]

    def get_rewards(self, name: str) -> RewardModel:
        """Gets the reward model with the specified name. Throws a RuntimeError if said model does not exist."""
        for model in self.rewards:
            if model.name == name:
                return model
        raise RuntimeError(f"Reward model {name} not present in model.")

    def get_states(self) -> dict[int, State]:
        return self.states

    def add_rewards(self, name: str) -> RewardModel:
        """Creates a reward model with the specified name and adds returns it."""
        for model in self.rewards:
            if model.name == name:
                raise RuntimeError(f"Reward model {name} already present in model.")
        reward_model = RewardModel(name, {})
        self.rewards.append(reward_model)
        return reward_model

    def get_observation(self, state: State) -> Observation:
        """Gets the observation for a given state."""
        if self.supports_observations and state.observation is not None:
            return self.states[state.id].get_observation()
        else:
            raise RuntimeError("Only POMDP models support observations")

    def get_rate(self, state: State) -> Number:
        """Gets the rate of a state."""
        if not self.supports_rates() or self.exit_rates is None:
            raise RuntimeError("Cannot get a rate of a deterministic-time model.")
        return self.exit_rates[state.id]

    def set_rate(self, state: State, rate: Number):
        """Sets the rate of a state."""
        if not self.supports_rates() or self.exit_rates is None:
            raise RuntimeError("Cannot set a rate of a deterministic-time model.")
        self.exit_rates[state.id] = rate

    def get_type(self):
        """Gets the type of this model"""
        return self.type

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

        if self.supports_rates() and self.exit_rates is not None:
            res += ["", "Exit rates:"] + [f"{self.exit_rates}"]

        if (
            self.supports_actions()
            and self.supports_rates()
            and self.markovian_states is not None
        ):
            markovian_states = [state.id for state in self.markovian_states]
            res += ["", "Markovian states:"] + [f"{markovian_states}"]

        return "\n".join(res)

    def __eq__(self, other):
        if isinstance(other, Model):
            return (
                self.type == other.type
                and self.states == other.states
                and self.transitions == other.transitions
                and sorted(self.rewards) == sorted(other.rewards)
                and self.exit_rates == other.exit_rates
                and self.markovian_states == other.markovian_states
                # TODO: and self.actions ==  other.actions
            )
        return False


def new_dtmc(name: str | None = None, create_initial_state: bool = True):
    """Creates a DTMC."""
    return Model(name, ModelType.DTMC, create_initial_state)


def new_mdp(name: str | None = None, create_initial_state: bool = True):
    """Creates an MDP."""
    return Model(name, ModelType.MDP, create_initial_state)


def new_ctmc(name: str | None = None, create_initial_state: bool = True):
    """Creates a CTMC."""
    return Model(name, ModelType.CTMC, create_initial_state)


def new_pomdp(name: str | None = None, create_initial_state: bool = True):
    """Creates a POMDP."""
    return Model(name, ModelType.POMDP, create_initial_state)


def new_ma(name: str | None = None, create_initial_state: bool = True):
    """Creates a MA."""
    return Model(name, ModelType.MA, create_initial_state)


def new_model(
    modeltype: ModelType, name: str | None = None, create_initial_state: bool = True
):
    """More general model creation function"""
    return Model(name, modeltype, create_initial_state)
