"""Contains the python model representations and their APIs."""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Tuple, cast

from stormvogel import parametric
import copy
import math

Number = float | Fraction | int
Value = Number | parametric.Parametric


def number_to_string(
    n: Value, use_fractions: bool, round_digits: int, denom_limit: int
) -> str:
    """Convert a Number to a string."""
    if isinstance(n, (int, float)):
        if math.isinf(float(n)):
            return "inf"
        if use_fractions:
            return str(Fraction(n).limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    elif isinstance(
        n, Fraction
    ):  # In the case of Fraction, a denominator of zero would have caused an error before.
        if use_fractions:
            return str(n.limit_denominator(denom_limit))
        else:
            return str(round(float(n), round_digits))
    elif isinstance(n, parametric.Parametric):
        return str(n)


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
        observation: the observation of a state as an integer
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
        valuations: The valuations of this state. Corresponds to Storm valuations/features.
        id: The id of this state.
        model: The model this state belongs to.
        observation: the observation of this state in case the model is a pomdp.
        name: the name of this state.
    """

    labels: list[str]
    valuations: dict[str, int | float | bool]
    id: int
    model: "Model"
    observation: Observation | None
    name: str

    def __init__(
        self,
        labels: list[str],
        valuations: dict[str, int | float | bool],
        id: int,
        model,
        name: str | None = None,
    ):
        self.model = model

        if id in self.model.states.keys():
            raise RuntimeError(
                "There is already a state with this id. Make sure the id is unique."
            )

        self.labels = labels
        self.valuations = valuations
        self.id = id
        self.observation = None

        # names must be unique
        if name is None:
            # if the user does not provide a name, we try to choose the id as a string
            name = str(id)
            if name in self.model.used_names:
                raise RuntimeError(
                    "You need to choose a state name because of a conflict (possibly because of state removal)."
                )
            self.model.used_names.add(name)
            self.name = name
        else:
            if name in self.model.used_names:
                raise RuntimeError(
                    "There is already a state with this name. Make sure the name is unique."
                )
            self.model.used_names.add(name)
            self.name = name

    def add_label(self, label: str):
        """adds a new label to the state"""
        if label in self.labels:
            raise RuntimeError(f"The label {label} is already present in this state.")

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

    def add_valuation(self, variable: str, value: int | bool | float):
        """Adds a valuation to the state."""
        self.valuations[variable] = value

    def available_actions(self) -> list["Action"]:
        """returns the list of all available actions in this state"""
        if self.model.supports_actions() and self.id in self.model.transitions.keys():
            action_list = []
            for action in self.model.transitions[self.id].transition.keys():
                action_list.append(action)
            return action_list
        else:
            return [EmptyAction]

    def get_outgoing_transitions(
        self, action: "Action | None" = None
    ) -> list[tuple[Value, "State"]] | None:
        """gets the outgoing transitions of this state"""

        # if the model supports actions we need to provide one
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

    def is_absorbing(self) -> bool:
        """returns if the state has a nonzero transition going to another state or not"""

        # for all actions we check if the state has outgoing transitions to a different state with value != 0
        for action in self.available_actions():
            transitions = self.get_outgoing_transitions(action)
            if transitions is not None:
                for transition in transitions:
                    assert isinstance(transition[0], (int, float))
                    if float(transition[0]) != 0 and transition[1] != self:
                        return False
        return True

    def is_initial(self):
        """Returns whether this state is initial."""
        return self == self.model.get_initial_state()

    def __str__(self):
        res = f"State {self.id} with labels {self.labels} and valuations {self.valuations}"
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
                    sorted(self.labels) == sorted(other.labels)
                    and observations_equal
                    and self.valuations == other.valuations
                )
        return False

    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return str(self.labels) < str(other.labels)


@dataclass(frozen=True)
class Action:
    """Represents an action, e.g., in MDPs.
        Note that this action object is completely independent of its corresponding branch.
        Their relation is managed by Transitions.
        Two actions with the same labels are considered equal.

    Args:
        labels: The labels of this action. Corresponds to Storm labels.
    """

    @staticmethod
    def create(labels: frozenset[str] | str | None = None) -> "Action":
        if isinstance(labels, str):
            return Action(frozenset({labels}))
        elif isinstance(labels, frozenset):
            return Action(labels)
        else:
            return Action(frozenset())

    labels: frozenset[str]

    def __lt__(self, other):
        if not isinstance(other, Action):
            return NotImplemented
        return str(self.labels) < str(other.labels)

    def __str__(self):
        return f"Action with labels {self.labels}"


# The empty action. Used for DTMCs and empty action transitions in mdps.
EmptyAction = Action(frozenset())


@dataclass(order=True)
class Branch:
    """Represents a branch, which is a distribution over states.

    Args:
        branch: The branch as a list of tuples.
            The first element is the probability value and the second element is the target state.
    """

    branch: list[tuple[Value, State]]

    def sort_states(self):
        """sorts the branch list by states"""
        self.branch.sort(key=lambda x: x[1])

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

    def sum_probabilities(self) -> Value:
        return sum([prob for (prob, _) in self.branch])  # type: ignore


class Transition:
    """Represents a transition, which map actions to branches.
        Note that an EmptyAction may be used if we want a non-action transition.
        Note that a single Transition might correspond to multiple 'arrows'.

    Args:
        transition: The transition dictionary. For each available action, we have a branch.
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

    def has_empty_action(self) -> bool:
        # Note that we don't have to deal with the corner case where there are both empty and non-empty transitions. This is dealt with at __init__.
        return self.transition.keys() == {EmptyAction}

    def __eq__(self, other):
        if isinstance(other, Transition):
            if len(self.transition) != len(other.transition):
                return False
            for action, other_action in zip(
                sorted(self.transition.keys()), sorted(other.transition.keys())
            ):
                if not (
                    action == other_action
                    and self.transition[action] == other.transition[action]
                ):
                    return False
            return True
        return False

    def sum_probabilities(self, action) -> Value:
        return self.transition[action].sum_probabilities()

    def is_stochastic(self, epsilon: Value) -> bool:
        """returns whether the probabilities in the branches sum to 1"""
        return all(
            [abs(self.sum_probabilities(a) - 1) <= epsilon for a in self.transition]  # type: ignore
        )

    def __getitem(self, item):
        return self.transition[item]


TransitionShorthand = list[tuple[Value, State]] | list[tuple[Action, State]]


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
        or isinstance(first_element, parametric.Parametric)
    ):
        return Transition(
            {EmptyAction: Branch(cast(list[tuple[Value, State]], shorthand))}
        )
    raise RuntimeError(
        f"Type of {first_element} not supported in transition {shorthand}"
    )


@dataclass()
class RewardModel:
    """Represents a state-exit reward model.
    Args:
        name: Name of the reward model.
        model: The model this rewardmodel belongs to.
        rewards: The rewards, the keys state action pairs.
    """

    name: str
    model: "Model"
    rewards: dict[Tuple[int, Action], Value]
    """Rewards dict. Hashed by state id and Action.
    The function update_rewards can be called to update rewards. After this, rewards will correspond to intermediate_rewards.
    Note that in models without actions, EmptyAction will be used here."""

    def __init__(
        self, name: str, model: "Model", rewards: dict[Tuple[int, Action], Value]
    ):
        self.name = name
        self.rewards = rewards
        self.model = model

    def set_from_rewards_vector(self, vector: list[Value]) -> None:
        """Set the rewards of this model according to a (stormpy) rewards vector."""
        combined_id = 0
        self.rewards = dict()
        for s in self.model.states.values():
            for a in s.available_actions():
                self.rewards[s.id, a] = vector[combined_id]
                combined_id += 1

    def get_state_reward(self, state: State) -> Value | None:
        """Gets the reward at said state or state action pair. Return None if no reward is present."""
        if self.model.supports_actions():
            raise RuntimeError(
                "This is a model with actions. Please call the get_state_action_reward(state, action) function instead"
            )
        if (state.id, EmptyAction) in self.rewards:
            return self.rewards[state.id, EmptyAction]
        else:
            return None

    def get_state_action_reward(self, state: State, action: Action) -> Value | None:
        """Gets the reward at said state or state action pair. Returns None if no reward was found."""
        if self.model.supports_actions():
            if action in state.available_actions():
                if (state.id, action) in self.rewards:
                    return self.rewards[state.id, action]
                else:
                    return None
            else:
                raise RuntimeError("This action is not available in this state")
        else:
            raise RuntimeError(
                "The model this rewardmodel belongs to does not support actions"
            )

    def set_state_reward(self, state: State, value: Value):
        """Sets the reward at said state. If the model has actions, try to use the empty state."""
        if self.model.supports_actions():
            self.set_state_action_reward(state, EmptyAction, value)
        else:
            self.rewards[state.id, EmptyAction] = value

    def set_state_action_reward(
        self,
        state: State,
        action: Action,
        value: Value,
        auto_update_rewards: bool = True,
    ):
        """sets the reward at said state action pair (in case of models with actions).
        If you disable auto_update_rewards, you will need to call update_intermediate_to"""
        if self.model.supports_actions():
            if action in state.available_actions():
                self.rewards[state.id, action] = value
            else:
                raise RuntimeError("This action is not available in this state")
        else:
            raise RuntimeError(
                "The model this rewardmodel belongs to does not support actions"
            )

    def get_reward_vector(self) -> list[Value]:
        """Return the rewards in a (stormpy) vector format."""
        vector = []
        for s in self.model.states.values():
            for a in s.available_actions():
                reward = self.rewards[s.id, a]
                if reward is None:
                    raise RuntimeError(
                        "A reward was not set. You might want to call set_unset_rewards."
                    )
                vector.append(reward)
        return vector

    def set_unset_rewards(self, value: Value):
        """Fills up rewards that were not set yet with the specified value.
        Use this if converting (to stormpy) doesn't work because the reward vector does not have the expected length."""
        for s in self.model.states.values():
            for a in s.available_actions():
                if (s.id, a) not in self.rewards:
                    self.rewards[s.id, a] = value

    def __lt__(self, other) -> bool:
        if not isinstance(other, RewardModel):
            return NotImplemented
        return self.name < other.name

    def __eq__(self, other) -> bool:
        if isinstance(other, RewardModel):
            return self.name == other.name and self.rewards == other.rewards
        return False


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
    actions: set[Action] | None
    rewards: list[RewardModel]
    # In ctmcs we work with rate transitions but additionally we can optionally store exit rates (hashed by id of the state)
    exit_rates: dict[int, Value] | None
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
            self.actions = set()
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

        # We also keep track of used state names
        self.used_names = set()

        # Add the initial state if specified to do so
        if create_initial_state:
            self.new_state(["init"])

    def summary(self):
        """Give a short summary of the model."""
        actions_bit = (
            f"{len(self.actions)} actions, " if self.actions is not None else ""
        )
        return (
            f"{self.type} model with name {self.name}, {len(self.get_states())} states, "
            + actions_bit
            + f"and {len(self.get_labels())} distinct labels."
        )

    def get_actions(self):
        """Return the actions of the model. Returns None if actions are not supported."""
        return self.actions

    def supports_actions(self):
        """Returns whether this model supports actions."""
        return self.get_type() in (ModelType.MDP, ModelType.POMDP, ModelType.MA)

    def supports_rates(self):
        """Returns whether this model supports rates."""
        return self.get_type() in (ModelType.CTMC, ModelType.MA)

    def supports_observations(self):
        """Returns whether this model supports observations."""
        return self.get_type() == ModelType.POMDP

    def is_parametric(self):
        """Returns whether this model contains parametric transition values"""
        for transition in self.transitions.values():
            for branch in transition.transition.values():
                for tup in branch.branch:
                    if isinstance(tup[0], parametric.Parametric):
                        return True
        return False

    def is_stochastic(self, epsilon: Value = 0.000001) -> bool:
        """For discrete models: Checks if all sums of outgoing transition probabilities for all states equal 1, with at most epsilon rounding error.
        For continuous models: Checks if all sums of outgoing rates sum to 0
        """

        if not self.supports_rates():
            return all(
                [
                    self.get_transitions(s).is_stochastic(epsilon)
                    for s in self.states
                    if s in self.transitions
                ]
            )

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
                    # we first calculate the sum
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

                    # then we divide each value by the sum
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

    def get_sub_model(self, states: list[State], normalize: bool = True) -> "Model":
        """Returns a submodel of the model based on a collection of states.
        The states in the collection are the states that stay in the model."""
        sub_model = copy.deepcopy(self)
        remove = []
        for state in sub_model.states.values():
            if state not in states:
                remove.append(state)
        for state in remove:
            sub_model.remove_state(state, normalize=False)

        if normalize:
            sub_model.normalize()
        return sub_model

    def parameter_valuation(self, values: dict[str, float]) -> "Model":
        """evaluates all parametric transitions with the given values and returns the induced model"""
        evaluated_model = copy.deepcopy(self)
        for state, transition in evaluated_model.transitions.items():
            for action, branch in transition.transition.items():
                new_branch = []
                for tup in branch.branch:
                    if isinstance(tup[0], parametric.Parametric):
                        tup = (tup[0].evaluate(values), tup[1])
                    new_branch.append(tup)
                evaluated_model.transitions[state].transition[
                    action
                ].branch = new_branch

        return evaluated_model

    def get_state_action_id(self, state: State, action: Action) -> int | None:
        """we calculate the appropriate state action id for a given state and action"""
        id = 0
        for s in self.states.values():
            for a in s.available_actions():
                if a == action and action in s.available_actions() and s == state:
                    return id
                id += 1

    def get_state_action_pair(self, id: int) -> tuple[State, Action] | None:
        """Given an id, we return the corresponding state action pair"""
        i = 0
        for s in self.states.values():
            for a in s.available_actions():
                if id == i:
                    return (s, a)
                i += 1

    def __free_state_id(self) -> int:
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

    def set_valuation_at_remaining_states(
        self, variables: list[str] | None = None, value: int | bool | float = 0
    ):
        """sets (dummy) value to variables in all states where they don't have a value yet"""

        # we either set it at all variables or just at a given subset of variables
        if variables is None:
            v = self.get_variables()
        else:
            v = variables

        # we set the values
        for state in self.states.values():
            for var in v:
                if var not in state.valuations.keys():
                    state.valuations[var] = value

    def has_unassigned_variables(self) -> bool:
        """we return whether this model has variables without a value"""
        # TODO return list of pairs of variables and states where it is undefined
        variables = self.get_variables()

        # if there are no variables at all, it is trivially true
        if variables == set():
            return False

        # we check all variables in all states
        for state in self.states.values():
            for variable in variables:
                if variable not in state.valuations.keys():
                    return True
        return False

    def all_states_outgoing_transition(self) -> bool:
        """checks if all states have an outgoing transition"""
        for state in self.states.items():
            if self.transitions.get(state[0]) is None:
                return False
        return True

    def add_markovian_state(self, markovian_state: State):
        """adds a state to the markovian states (in case of markov automatas)"""
        if self.get_type() == ModelType.MA and self.markovian_states is not None:
            self.markovian_states.append(markovian_state)
        else:
            raise RuntimeError("This model is not a MA")

    def set_transitions(
        self, s: State, transitions: Transition | TransitionShorthand
    ) -> None:
        """Set the transition from a state."""
        if not isinstance(transitions, Transition):
            transitions = transition_from_shorthand(transitions)
        if self.actions is not None and EmptyAction in transitions.transition.keys():
            self.actions.add(EmptyAction)
        self.transitions[s.id] = transitions

    def add_transitions(
        self, s: State, transitions: Transition | TransitionShorthand
    ) -> None:
        """Add new transitions from a state to the model. If no transition currently exists, the result will be the same as set_transitions."""

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
                for action, branch in transitions.transition.items():
                    assert self.actions is not None
                    if action not in self.actions:
                        self.actions.add(action)
                    self.transitions[s.id].transition[action] = branch

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

    def get_action_with_labels(self, labels: frozenset[str]) -> Action | None:
        """Get the action with provided list of labels"""
        assert self.actions is not None
        for action in self.actions:
            if action.labels == labels:
                return action

    def new_action(self, labels: frozenset[str] | str | None = None) -> Action:
        """Creates a new action and returns it."""
        if not self.supports_actions():
            raise RuntimeError(
                "Called new_action on a model that does not support actions"
            )
        assert self.actions is not None
        action = Action.create(labels)
        self.actions.add(action)
        return action

    def reassign_ids(self):
        """Reassigns the ids of states, transitions and rates to be in order again.
        Mainly useful to keep consistent with storm."""

        print(
            "Warning: Using this can cause problems in your code if there are existing references to states by id."
        )

        # we change the ids in the dictionaries of the model object
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

        # we change the ids in the states themselves
        for index, state in enumerate(self.states.values()):
            state.id = index

    def remove_state(
        self, state: State, normalize: bool = True, reassign_ids: bool = False
    ):
        """Properly removes a state, it can optionally normalize the model and reassign ids automatically."""

        if state in self.states.values():
            # we remove the state from the transitions
            # first we remove transitions that go into the state
            remove_actions_index = []
            for index, transition in self.transitions.items():
                for action, branch in transition.transition.items():
                    for index_tuple, tuple in enumerate(branch.branch):
                        # remove the tuple if it refernces the state
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
        else:
            raise RuntimeError("This state is not part of this model.")

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

    def action(self, labels: frozenset[str] | str | None) -> Action:
        """New action or get action if it exists."""
        if not self.supports_actions():
            raise RuntimeError(
                "Called method action on a model that does not support actions"
            )
        assert self.actions is not None
        action = Action.create(labels)

        if action not in self.actions:
            self.new_action(labels)
        return action

    def new_state(
        self,
        labels: list[str] | str | None = None,
        valuations: dict[str, int | bool | float] | None = None,
        name: str | None = None,
        id: int | None = None,
    ) -> State:
        """Creates a new state and returns it."""

        # we can either provide an id, or check which one is free
        if id is None:
            state_id = self.__free_state_id()
        else:
            state_id = id

        if isinstance(labels, list):
            state = State(labels, valuations or {}, state_id, self, name=name)
        elif isinstance(labels, str):
            state = State([labels], valuations or {}, state_id, self, name=name)
        elif labels is None:
            state = State([], valuations or {}, state_id, self, name=name)

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

    def get_state_by_id(self, state_id: int) -> State:
        """Get a state by its id."""
        if state_id not in self.states:
            raise RuntimeError("Requested a non-existing state")
        return self.states[state_id]

    def get_state_by_name(self, state_name) -> State | None:
        """Get a state by its name."""
        names = {state.name: state for state in self.states.values()}
        if state_name not in names:
            raise RuntimeError("Requested a non-existing state")

        return names[state_name]

    def get_initial_state(self) -> State:
        """Gets the initial state (id=0)."""
        return self.states[0]

    def get_ordered_labels(self) -> list[list[str]]:
        """Get all the labels of this model, ordered by id.
        IMPORTANT: If a state has no label, then a value '' is inserted!"""
        return [
            (s.labels if len(s.labels) > 0 else []) for s in self.get_states().values()
        ]

    def get_labels(self) -> set[str]:
        """Get all labels in states of this Model."""
        collected_labels: set[str] = set()
        for _id, state in self.states.items():
            collected_labels = collected_labels | set(state.labels)
        return collected_labels

    def get_variables(self) -> set[str]:
        """gets the set of all variables present in this model (features)"""
        variables: set[str] = set()
        for _id, state in self.states.items():
            variables = variables | set(state.valuations.keys())
        return variables

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

    def get_parameters(self) -> set[str]:
        """Returns the set of parameters of this model"""
        parameters = set()
        for transition in self.transitions.values():
            for branch in transition.transition.values():
                for tup in branch.branch:
                    if isinstance(tup[0], parametric.Parametric):
                        parameters = parameters.union(tup[0].get_variables())
        return parameters

    def get_states(self) -> dict[int, State]:
        return self.states

    def add_rewards(self, name: str) -> RewardModel:
        """Creates a reward model with the specified name, adds it and returns it."""
        for model in self.rewards:
            if model.name == name:
                raise RuntimeError(f"Reward model {name} already present in model.")
        reward_model = RewardModel(name, self, {})
        self.rewards.append(reward_model)
        return reward_model

    def get_observation(self, state: State) -> Observation:
        """Gets the observation for a given state."""
        if self.supports_observations and state.observation is not None:
            return self.states[state.id].get_observation()
        else:
            raise RuntimeError("Only POMDP models support observations")

    def get_rate(self, state: State) -> Value:
        """Gets the rate of a state."""
        if not self.supports_rates() or self.exit_rates is None:
            raise RuntimeError("Cannot get a rate of a deterministic-time model.")
        return self.exit_rates[state.id]

    def set_rate(self, state: State, rate: Value):
        """Sets the rate of a state."""
        if not self.supports_rates() or self.exit_rates is None:
            raise RuntimeError("Cannot set a rate of a deterministic-time model.")
        self.exit_rates[state.id] = rate

    def get_type(self) -> ModelType:
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
                    dot += f'{state_id} [ label = "", shape=point ];\n'
        for state_id, transition in self.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    # Only draw probabilities
                    for prob, target in branch.branch:
                        dot += f'{state_id} -> {target.id} [ label = "{prob}" ];\n'
                else:
                    # Draw actions, then probabilities
                    dot += f'{state_id} -> {state_id} [ label = "{action.labels}" ];\n'
                    for prob, target in branch.branch:
                        dot += f'{state_id} -> {target.id} [ label = "{prob}" ];\n'

        dot += "}"
        return dot

    def __str__(self) -> str:
        res = [f"{self.type} with name {self.name}"]
        res += ["", "States:"] + [f"{state}" for (_id, state) in self.states.items()]
        res += ["", "Transitions:"] + [
            f"{id}: {transition}" for (id, transition) in self.transitions.items()
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

    def __eq__(self, other) -> bool:
        if isinstance(other, Model):
            return (
                self.actions == other.actions
                and self.type == other.type
                and self.states == other.states
                and self.transitions == other.transitions
                and sorted(self.rewards) == sorted(other.rewards)
                and self.exit_rates == other.exit_rates
                and self.markovian_states == other.markovian_states
            )
        return False

    def __getitem__(self, state_id: int):
        return self.states[state_id]


def new_dtmc(name: str | None = None, create_initial_state: bool = True) -> Model:
    """Creates a DTMC."""
    return Model(name, ModelType.DTMC, create_initial_state)


def new_mdp(name: str | None = None, create_initial_state: bool = True) -> Model:
    """Creates an MDP."""
    return Model(name, ModelType.MDP, create_initial_state)


def new_ctmc(name: str | None = None, create_initial_state: bool = True) -> Model:
    """Creates a CTMC."""
    return Model(name, ModelType.CTMC, create_initial_state)


def new_pomdp(name: str | None = None, create_initial_state: bool = True) -> Model:
    """Creates a POMDP."""
    return Model(name, ModelType.POMDP, create_initial_state)


def new_ma(name: str | None = None, create_initial_state: bool = True) -> Model:
    """Creates a MA."""
    return Model(name, ModelType.MA, create_initial_state)


def new_model(
    modeltype: ModelType, name: str | None = None, create_initial_state: bool = True
) -> Model:
    """More general model creation function"""
    return Model(name, modeltype, create_initial_state)
