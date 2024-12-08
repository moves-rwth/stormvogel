from stormvogel import pgc
import stormvogel.model
import math
import stormvogel.mapping


def test_pgc_mdp():
    # we build the model with pgc:
    N = 2
    p = 0.5
    initial_state = pgc.State(x=math.floor(N / 2))

    left = pgc.Action(["left"])
    right = pgc.Action(["right"])

    def available_actions(s: pgc.State):
        return [left, right]

    def rewards(s: pgc.State, a: pgc.Action):
        return 1

    def labels(s: pgc.State):
        return [str(s.x)]

    def delta(s: pgc.State, action: pgc.Action):
        if action == left:
            return (
                [
                    (p, pgc.State(x=s.x + 1)),
                    (1 - p, pgc.State(x=s.x)),
                ]
                if s.x < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, pgc.State(x=s.x - 1)),
                    (1 - p, pgc.State(x=s.x)),
                ]
                if s.x > 0
                else []
            )

    pgc_model = stormvogel.pgc.build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
        labels=labels,
        rewards=rewards,
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

    rewardmodel = model.add_rewards("rewards")
    for i in range(2 * N):
        rewardmodel.set_state_action_reward_at_id(i, 1)

    assert model == pgc_model


def test_pgc_dtmc():
    # we build the model with pgc:
    p = 0.5
    initial_state = pgc.State(s=0)

    def rewards(s: pgc.State):
        return 1

    def delta(s: pgc.State):
        match s.s:
            case 0:
                return [(p, pgc.State(s=1)), (1 - p, pgc.State(s=2))]
            case 1:
                return [(p, pgc.State(s=3)), (1 - p, pgc.State(s=4))]
            case 2:
                return [(p, pgc.State(s=5)), (1 - p, pgc.State(s=6))]
            case 3:
                return [(p, pgc.State(s=1)), (1 - p, pgc.State(s=7, d=1))]
            case 4:
                return [
                    (p, pgc.State(s=7, d=2)),
                    (1 - p, pgc.State(s=7, d=3)),
                ]
            case 5:
                return [
                    (p, pgc.State(s=7, d=4)),
                    (1 - p, pgc.State(s=7, d=5)),
                ]
            case 6:
                return [(p, pgc.State(s=2)), (1 - p, pgc.State(s=7, d=6))]
            case 7:
                return [(1, pgc.State(s=7))]

    pgc_model = stormvogel.pgc.build_pgc(
        delta=delta,
        initial_state_pgc=initial_state,
        rewards=rewards,
        modeltype=stormvogel.model.ModelType.DTMC,
    )

    # we build the model in the regular way:
    model = stormvogel.model.new_dtmc()
    model.states[0].features = {"s": 0}
    model.set_transitions(
        model.get_initial_state(),
        [
            (1 / 2, model.new_state(features={"s": 1})),
            (1 / 2, model.new_state(features={"s": 2})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(1),
        [
            (1 / 2, model.new_state(features={"s": 3})),
            (1 / 2, model.new_state(features={"s": 4})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(2),
        [
            (1 / 2, model.new_state(features={"s": 5})),
            (1 / 2, model.new_state(features={"s": 6})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(3),
        [
            (1 / 2, model.get_state_by_id(1)),
            (1 / 2, model.new_state(features={"s": 7, "d": 1})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(4),
        [
            (1 / 2, model.new_state(features={"s": 7, "d": 2})),
            (1 / 2, model.new_state(features={"s": 7, "d": 3})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(5),
        [
            (1 / 2, model.new_state(features={"s": 7, "d": 4})),
            (1 / 2, model.new_state(features={"s": 7, "d": 5})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(6),
        [
            (1 / 2, model.get_state_by_id(2)),
            (1 / 2, model.new_state(features={"s": 7, "d": 6})),
        ],
    )
    model.set_transitions(
        model.get_state_by_id(7), [(1, model.new_state(features={"s": 7}))]
    )
    model.set_transitions(model.get_state_by_id(8), [(1, model.get_state_by_id(13))])
    model.set_transitions(model.get_state_by_id(9), [(1, model.get_state_by_id(13))])
    model.set_transitions(model.get_state_by_id(10), [(1, model.get_state_by_id(13))])
    model.set_transitions(model.get_state_by_id(11), [(1, model.get_state_by_id(13))])
    model.set_transitions(model.get_state_by_id(12), [(1, model.get_state_by_id(13))])
    model.set_transitions(model.get_state_by_id(13), [(1, model.get_state_by_id(13))])

    rewardmodel = model.add_rewards("rewards")
    for state in model.states.values():
        rewardmodel.set_state_reward(state, 1)

    assert pgc_model == model
