from stormvogel.visualization import Visualization
from stormvogel.model import Model, ModelType


def test_visualization(mocker):
    class MockNetwork:
        add_node = mocker.stub(name="add_node_stub")
        add_edge = mocker.stub(name="add_edge_stub")
        set_options = mocker.stub(name="set_options_stub")

    model = Model("simple", ModelType.MDP)
    model.new_state("one")
    vis = Visualization(model)
    vis.nt = MockNetwork
    vis.prepare()
    MockNetwork.add_node.assert_any_call(
        0, label="init", group="states", position_dict={}
    )
    MockNetwork.add_node.assert_any_call(
        1, label="one", group="states", position_dict={}
    )
    assert MockNetwork.add_node.call_count == 2
