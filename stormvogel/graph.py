from enum import Enum
from typing import Self
from networkx import DiGraph

from stormvogel.model import Action, EmptyAction, Model

ACTION_ID_OFFSET = 10**10


class NodeType(Enum):
    STATE = 0
    ACTION = 1
    UNDEFINED = 2


class ModelGraph(DiGraph):
    @classmethod
    def from_model(cls, model: Model) -> Self:
        """Construct a directed graph from a model instance"""
        G = cls()
        state_action_id_map: dict[tuple[int, Action], int] = (
            dict()
        )  # state_id, action -> action id
        action_id = ACTION_ID_OFFSET
        for state in model.states.values():
            G.add_node(
                state.id, type_=NodeType.STATE
            )  # TODO: other params like reward, obs

        for state_id, transition in model.transitions.items():
            for action, branch in transition.transition.items():
                if action == EmptyAction:
                    for probability, target in branch.branch:
                        G.add_edge(state_id, target.id, probability=probability)
                        # TODO: we might want a node here
                    continue
                G.add_node(action_id, type_=NodeType.ACTION)
                G.add_edge(state_id, action_id)
                for probability, target in branch.branch:
                    G.add_edge(action_id, target.id, probability=probability)
                state_action_id_map[state_id, action] = action_id
                action_id += 1
        return G
