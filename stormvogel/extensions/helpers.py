from bidict import bidict
from typing import Any
import stormpy


def choice_mapping(sv_model, sp_model):
    """Return a bijective mapping between the stormvogel state-action pairs and the stormvogel model.
    WARNING: This function will be depricated later. It might also be faulty, I don't know :))"""
    res = bidict({})
    choice_id = 0
    for s in sv_model.states.values():
        for a in s.available_actions():
            res[s.id, a] = choice_id
            choice_id += 1
    return res


def to_bit_vector(state_set: set[int], model: Any):
    return stormpy.BitVector(model.transition_matrix.nr_columns, list(state_set))
