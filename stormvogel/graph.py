from collections.abc import Callable
from enum import Enum
from typing import Any, Self
from networkx import DiGraph

from stormvogel.model import Action, EmptyAction, Model, State, Value

ACTION_ID_OFFSET = 10**10


class NodeType(Enum):
    STATE = 0
    ACTION = 1
    UNDEFINED = 2


class ModelGraph(DiGraph):
    def __init__(
        self,
        model: Model,
        state_properties: Callable[[State], dict[str, Any]] | None = None,
        action_properties: Callable[[State, Action], dict[str, Any]] | None = None,
        transition_properties: Callable[[State, Action, State], dict[str, Any]]
        | None = None,
    ) -> None:
        super().__init__(self)
        self.model = model
        self._state_action_id_map: dict[tuple[int, Action], int] = dict()
        self._next_action_id = ACTION_ID_OFFSET
        self._build_from_model(
            state_properties, action_properties, transition_properties
        )

    @property
    def state_action_id_map(self):
        return self._state_action_id_map

    def add_state(self, state: State | int, **attr):
        if isinstance(state, State):
            state = state.id
        self.add_node(state, type=NodeType.STATE, **attr)

    def add_action(self, state: State | int, action: Action, **action_attr):
        if isinstance(state, State):
            state = state.id
        assert state in self.nodes, f"State {state} not in graph yet"
        if action == EmptyAction:
            return
        self.add_node(self._next_action_id, type=NodeType.ACTION, **action_attr)
        self.add_edge(state, self._next_action_id)
        self._state_action_id_map[(state, action)] = self._next_action_id
        self._next_action_id += 1

    def add_transition(
        self,
        state: State | int,
        action: Action,
        next_state: int | State,
        probability: Value,
        **attr,
    ):
        if isinstance(state, State):
            state = state.id
        if isinstance(next_state, State):
            next_state = next_state.id
        action_id = self._state_action_id_map.get((state, action), None)
        assert state in self.nodes
        assert action == EmptyAction or action_id is not None
        assert next_state in self.nodes
        if action == EmptyAction:
            self.add_edge(state, next_state, probability=probability, **attr)
        else:
            self.add_edge(action_id, next_state, probability=probability, **attr)

    def _build_from_model(
        self,
        state_properties: Callable[[State], dict[str, Any]] | None = None,
        action_properties: Callable[[State, Action], dict[str, Any]] | None = None,
        transition_properties: Callable[[State, Action, State], dict[str, Any]]
        | None = None,
    ) -> Self:
        """Construct a directed graph from a model instance"""
        for state in self.model.states.values():
            props = dict()
            if state_properties is not None:
                props = state_properties(state)
            self.add_state(state, **props)

        for state_id, transition in self.model.transitions.items():
            state = self.model.get_state_by_id(state_id)
            for action, branch in transition.transition.items():
                action_props = dict()
                if action_properties is not None:
                    action_props = action_properties(state, action)
                self.add_action(state_id, action, **action_props)
                for probability, target in branch.branch:
                    transition_props = dict()
                    if transition_properties is not None:
                        transition_props = transition_properties(state, action, target)
                    self.add_transition(
                        state_id,
                        action,
                        target,
                        probability=probability,
                        **transition_props,
                    )
        return self
