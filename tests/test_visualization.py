from stormvogel.visualization import Visualization
from stormvogel.model import Model, ModelType


def test_show(mocker):
    class MockNetwork:
        def __init__(self, *args, **kwargs):
            self.init(*args, **kwargs)

        init = mocker.stub(name="init_stub")
        add_node = mocker.stub(name="add_node_stub")
        add_edge = mocker.stub(name="add_edge_stub")
        set_options = mocker.stub(name="set_options_stub")
        show = mocker.stub(name="show_stub")

    mocker.patch("stormvogel.visjs.Network", MockNetwork)
    model = Model("simple", ModelType.MDP)
    one = model.new_state("one")
    init = model.get_initial_state()
    model.set_transitions(init, [(1, one)])
    vis = Visualization(model)
    vis.show()
    MockNetwork.add_node.assert_any_call(
        0, label="init", group="states", position_dict={}
    )  # type: ignore
    MockNetwork.add_node.assert_any_call(
        1, label="one", group="states", position_dict={}
    )  # type: ignore
    assert MockNetwork.add_node.call_count == 2
    MockNetwork.add_edge.assert_any_call(0, 1, label="1")
    assert MockNetwork.add_edge.call_count == 1


def test_rewards(mocker):
    class MockNetwork:
        def __init__(self, *args, **kwargs):
            self.init(*args, **kwargs)

        init = mocker.stub(name="init_stub")
        add_node = mocker.stub(name="add_node_stub")
        add_edge = mocker.stub(name="add_edge_stub")
        set_options = mocker.stub(name="set_options_stub")
        show = mocker.stub(name="show_stub")

    mocker.patch("stormvogel.visjs.Network", MockNetwork)
    model = Model("simple", ModelType.MDP)
    one = model.new_state("one")
    init = model.get_initial_state()
    model.set_transitions(init, [(1, one)])
    model.add_rewards("LOL")
    model.get_rewards("LOL").set(one, 37)
    vis = Visualization(model=model)
    vis.show()
    MockNetwork.add_node.assert_any_call(
        0, label="init", group="states", position_dict={}
    )  # type: ignore
    MockNetwork.add_node.assert_any_call(
        1, label="one\nLOL: 37", group="states", position_dict={}
    )  # type: ignore
    assert MockNetwork.add_node.call_count == 2
    MockNetwork.add_edge.assert_any_call(0, 1, label="1")
    assert MockNetwork.add_edge.call_count == 1
