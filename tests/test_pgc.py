from stormvogel import pgc
import stormvogel.model
import math
import stormpy
import stormvogel.mapping


def test_pgc_mdp():
    # we build the model with pgc:
    N = 2
    p = 0.5
    initial_state = pgc.State({"x": math.floor(N / 2)})

    left = pgc.Action(["left"])
    right = pgc.Action(["right"])

    def available_actions(s: pgc.State):
        return [left, right]

    def delta(s: pgc.State, action: pgc.Action):
        if action == left:
            print(s.f)
            print(s.f["x"])
            return (
                [
                    (p, pgc.State({"x": s.f["x"] + 1})),
                    (1 - p, pgc.State({"x": s.f["x"]})),
                ]
                if s.f["x"] < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, pgc.State({"x": s.f["x"] - 1})),
                    (1 - p, pgc.State({"x": s.f["x"]})),
                ]
                if s.f["x"] > 0
                else []
            )

    pgc_model = stormvogel.pgc.build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
    )

    # we build the model in the regular way:
    model = stormvogel.model.new_mdp(create_initial_state=False)
    state1 = model.new_state(labels=["init"], features={"x": 1})
    state2 = model.new_state(features={"x": 2})
    state0 = model.new_state(features={"0": 0})
    left = model.new_action("left", frozenset({"left"}))
    right = model.new_action("right", frozenset({"right"}))
    branch11 = stormvogel.model.Branch([(0.5, state1), (0.5, state2)])
    branch12 = stormvogel.model.Branch([(0.5, state1), (0.5, state0)])
    branch0 = stormvogel.model.Branch([(0.5, state0), (0.5, state1)])
    branch2 = stormvogel.model.Branch([(0.5, state2), (0.5, state1)])

    model.add_transitions(
        state1, stormvogel.model.Transition({left: branch11, right: branch12})
    )
    model.add_transitions(state2, stormvogel.model.Transition({right: branch2}))
    model.add_transitions(state0, stormvogel.model.Transition({left: branch0}))

    print(model)
    print(pgc_model)

    assert model == pgc_model


def test_pgc_dtmc():
    # we build the model with pgc:
    p = 0.5
    initial_state = pgc.State({"s": 0})

    def delta(s: pgc.State):
        match s.f["s"]:
            case 0:
                return [(p, pgc.State({"s": 1})), (1 - p, pgc.State({"s": 2}))]
            case 1:
                return [(p, pgc.State({"s": 3})), (1 - p, pgc.State({"s": 4}))]
            case 2:
                return [(p, pgc.State({"s": 5})), (1 - p, pgc.State({"s": 6}))]
            case 3:
                return [(p, pgc.State({"s": 1})), (1 - p, pgc.State({"s": 7, "d": 1}))]
            case 4:
                return [
                    (p, pgc.State({"s": 7, "d": 2})),
                    (1 - p, pgc.State({"s": 7, "d": 3})),
                ]
            case 5:
                return [
                    (p, pgc.State({"s": 7, "d": 4})),
                    (1 - p, pgc.State({"s": 7, "d": 5})),
                ]
            case 6:
                return [(p, pgc.State({"s": 2})), (1 - p, pgc.State({"s": 7, "d": 6}))]
            case 7:
                return [(1, pgc.State({"s": 7}))]

    pgc_model = stormvogel.pgc.build_pgc(
        delta=delta,
        initial_state_pgc=initial_state,
        modeltype=stormvogel.model.ModelType.DTMC,
    )

    # we build the model in the regular way:
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)
    print(dir(model.states[0]))
    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(model)
    print(pgc_model)
    print(stormvogel_model)
