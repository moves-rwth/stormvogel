from stormvogel import pgc
import stormvogel.model
import math


def test_pgc_mdp():
    # we build the model with pgc:
    N = 2
    p = 0.5
    initial_state = pgc.State(math.floor(N / 2))

    left = pgc.Action(["left"])
    right = pgc.Action(["right"])

    def available_actions(s: pgc.State):
        return [left, right]

    def delta(s: pgc.State, action: pgc.Action):
        if action == left:
            return (
                [(p, pgc.State(x=s.x + 1)), (1 - p, pgc.State(x=s.x))]
                if s.x < N
                else []
            )
        elif action == right:
            return (
                [(p, pgc.State(x=s.x - 1)), (1 - p, pgc.State(x=s.x))]
                if s.x > 0
                else []
            )

    pgc_model = stormvogel.pgc.build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
    )

    # we build the model in the regular way:
    model = stormvogel.model.new_mdp(create_initial_state=False)
    state1 = model.new_state(labels=["init", "1"], features={"x": 1})
    state2 = model.new_state(labels=["2"], features={"x": 2})
    state0 = model.new_state(labels=["0"], features={"0": 0})
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

    assert model == pgc_model
