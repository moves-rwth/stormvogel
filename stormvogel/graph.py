from enum import Enum
from typing import Self
from networkx import DiGraph

from stormvogel.model import Action, EmptyAction, Model, State, Value

ACTION_ID_OFFSET = 10**10


class NodeType(Enum):
    STATE = 0
    ACTION = 1
    UNDEFINED = 2


class ModelGraph(DiGraph):
    def __init__(self, incoming_graph_data=None, **attr) -> None:
        super().__init__(incoming_graph_data, **attr)
        self.state_action_id_map: dict[tuple[int, Action], int] = dict()
        self._next_action_id = ACTION_ID_OFFSET

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
        self.state_action_id_map[(state, action)] = self._next_action_id
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
        action_id = self.state_action_id_map.get((state, action), None) is not None
        assert state in self.nodes
        assert action == EmptyAction or action_id is not None
        assert next_state in self.nodes
        if action == EmptyAction:
            self.add_edge(state, next_state, probability=probability, **attr)
        else:
            self.add_edge(action_id, next_state, probability=probability, **attr)

    @classmethod
    def from_model(cls, model: Model) -> Self:
        """Construct a directed graph from a model instance"""
        G = cls()
        for state in model.states.values():
            G.add_state(
                state, type_=NodeType.STATE
            )  # TODO: other params like reward, obs

        for state_id, transition in model.transitions.items():
            for action, branch in transition.transition.items():
                G.add_action(state_id, action)
                for probability, target in branch.branch:
                    G.add_transition(state_id, action, target, probability=probability)
        return G
