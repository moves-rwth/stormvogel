from bidict import bidict
from typing import Any

try:
    import stormpy
except ImportError:
    stormpy = None


def choice_mapping(sv_model, sp_model):
    """Return a bijective mapping between the stormvogel state-action pairs and the stormvogel model.
    WARNING: This function will be depricated later. It might also be faulty, I don't know :))"""
    res = bidict({})
    choice_id = 0
    for _, s in sv_model:
        for a in s.available_actions():
            res[s.id, a] = choice_id
            choice_id += 1
    return res


def to_bit_vector(state_set: set[int], model: Any):
    assert stormpy is not None
    return stormpy.BitVector(model.transition_matrix.nr_columns, list(state_set))
