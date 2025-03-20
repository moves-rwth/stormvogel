from stormvogel.visualization import Visualization
from stormvogel.model import Model, ModelType
from stormvogel.result import Result, Scheduler


def boilerplate(mocker):
    class MockNetwork:
        def __init__(self, *args, **kwargs):
            self.init(*args, **kwargs)

        init = mocker.stub(name="init_stub")
        add_node = mocker.stub(name="add_node_stub")
        add_edge = mocker.stub(name="add_edge_stub")
        set_options = mocker.stub(name="set_options_stub")
        show = mocker.stub(name="show_stub")

    mocker.patch("stormvogel.visjs.Network", MockNetwork)
    return MockNetwork


def simple_model():
    model = Model("simple", ModelType.DTMC)
    one = model.new_state("one")
    init = model.get_initial_state()
    model.set_transitions(init, [(1, one)])
    return model, one, init


def test_show(mocker):
    MockNetwork = boilerplate(mocker)
    model, one, init = simple_model()
    vis = Visualization(model)
    vis.show()
    MockNetwork.init.assert_any_call(
        name=vis.name,
        width=vis.layout.layout["misc"]["width"],
        height=vis.layout.layout["misc"]["height"],
        output=vis.output,
        debug_output=vis.debug_output,
        do_display=False,
        do_init_server=vis.do_init_server,
        positions=vis.layout.layout["positions"],
        use_iframe=False,
    )
    MockNetwork.add_node.assert_any_call(0, label="init", group="states")  # type: ignore
    MockNetwork.add_node.assert_any_call(1, label="one", group="states")  # type: ignore
    assert MockNetwork.add_node.call_count == 2
    MockNetwork.add_edge.assert_any_call(0, 1, label="1")
    assert MockNetwork.add_edge.call_count == 1


def test_rewards(mocker):
    MockNetwork = boilerplate(mocker)
    model, one, init = simple_model()
    model.set_transitions(init, [(1, one)])
    model.add_rewards("LOL")
    model.get_rewards("LOL").set_state_reward(one, 37)
    model.add_rewards("HIHI")
    model.get_rewards("HIHI").set_state_reward(one, 42)
    vis = Visualization(model=model)
    vis.show()
    MockNetwork.add_node.assert_any_call(0, label="init", group="states")  # type: ignore
    MockNetwork.add_node.assert_any_call(
        1, label="one\n€\tLOL: 37\tHIHI: 42", group="states"
    )  # type: ignore
    assert MockNetwork.add_node.call_count == 2
    MockNetwork.add_edge.assert_any_call(0, 1, label="1")
    assert MockNetwork.add_edge.call_count == 1


def test_results_count(mocker):
    MockNetwork = boilerplate(mocker)
    model, one, init = simple_model()
    result = Result(model, {0: 69, 1: 12})

    vis = Visualization(model=model, result=result)
    vis.show()
    RES_SYM = vis.layout.layout["state_properties"]["result_symbol"]
    MockNetwork.add_node.assert_any_call(0, label=f"init\n{RES_SYM} 69", group="states")  # type: ignore
    MockNetwork.add_node.assert_any_call(1, label=f"one\n{RES_SYM} 12", group="states")  # type: ignore

    assert result.values == {0: 69, 1: 12}
    assert MockNetwork.add_node.call_count == 2
    MockNetwork.add_edge.assert_any_call(0, 1, label="1")
    assert MockNetwork.add_edge.call_count == 1


def test_results_scheduler(mocker):
    MockNetwork = boilerplate(mocker)
    model = Model("mdp", model_type=ModelType.MDP)
    init = model.get_initial_state()
    good = model.new_action(frozenset(["GOOD"]))
    bad = model.new_action(frozenset(["BAD"]))
    end = model.new_state("end")
    model.set_transitions(init, [(good, end), (bad, end)])
    scheduler = Scheduler(model, {0: good})
    result = Result(model, {0: 1, 1: 2}, scheduler)
    vis = Visualization(model=model, result=result)
    vis.show()
    MockNetwork.add_node.assert_any_call(id=10000000001, label="BAD", group="actions")
    MockNetwork.add_node.assert_any_call(
        id=10000000000, label="GOOD", group="scheduled_actions"
    )

    assert MockNetwork.add_node.call_count == 4
    assert MockNetwork.add_edge.call_count == 4
