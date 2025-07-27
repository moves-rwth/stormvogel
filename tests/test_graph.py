import pytest

import stormvogel.examples as examples
from stormvogel.graph import ModelGraph


@pytest.mark.parametrize(
    "model",
    [
        examples.create_monty_hall_mdp(),
        examples.create_monty_hall_pomdp(),
        examples.create_lion_mdp(),
        examples.create_car_mdp(),
        examples.create_die_dtmc(),
        examples.create_nuclear_fusion_ctmc(),
        examples.create_study_mdp(),
    ],
)
def test_graph_creation(model):
    G = ModelGraph.from_model(model)
    assert all(state.id in G.nodes for state in model.states.values()), (
        "Missing state in ModelGraph"
    )
