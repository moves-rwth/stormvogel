try:
    import stormpy
except ImportError:
    stormpy = None

import stormvogel.stormpy_utils.mapping


def from_prism(prism_code="stormpy.storage.storage.PrismProgram"):
    """Create a model from prism. Requires stormpy."""

    assert stormpy is not None
    return stormvogel.mapping.stormpy_to_stormvogel(stormpy.build_model(prism_code))
