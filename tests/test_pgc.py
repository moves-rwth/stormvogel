from stormvogel import pgc, model
import math


def test_pgc_mdp():
    # we build the model with pgc:
    N = 2
    p = 0.5
    initial_state = pgc.State(x=math.floor(N / 2))

    left = pgc.Action(["left"])
    right = pgc.Action(["right"])

    def available_actions(s: pgc.State):
        if s.x == N:
            return [right]
        elif s.x == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s: pgc.State, a: pgc.Action):
        return [1, 2]

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

    pgc_model = pgc.build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
        labels=labels,
        rewards=rewards,
    )

    # we build the model in the regular way:
    regular_model = model.new_mdp(create_initial_state=False)
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    left = regular_model.new_action(frozenset({"left"}))
    right = regular_model.new_action(frozenset({"right"}))
    branch12 = model.Branch([(0.5, state1), (0.5, state2)])
    branch10 = model.Branch([(0.5, state1), (0.5, state0)])
    branch01 = model.Branch([(0.5, state0), (0.5, state1)])
    branch21 = model.Branch([(0.5, state2), (0.5, state1)])

    regular_model.add_transitions(
        state1, model.Transition({left: branch12, right: branch10})
    )
    regular_model.add_transitions(state2, model.Transition({right: branch21}))
    regular_model.add_transitions(state0, model.Transition({left: branch01}))

    rewardmodel = regular_model.add_rewards("rewardmodel: 0")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.add_rewards("rewardmodel: 1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    assert regular_model == pgc_model


def test_pgc_mdp_int():
    # we build the model with pgc:
    N = 2
    p = 0.5
    initial_state = math.floor(N / 2)

    left = pgc.Action(["left"])
    right = pgc.Action(["right"])

    def available_actions(s):
        if s == N:
            return [right]
        elif s == 0:
            return [left]
        else:
            return [left, right]

    def rewards(s, a: pgc.Action):
        return [1, 2]

    def labels(s):
        return [str(s)]

    def delta(s, action: pgc.Action):
        if action == left:
            return (
                [
                    (p, s + 1),
                    (1 - p, s),
                ]
                if s < N
                else []
            )
        elif action == right:
            return (
                [
                    (p, s - 1),
                    (1 - p, s),
                ]
                if s > 0
                else []
            )

    pgc_model = pgc.build_pgc(
        delta=delta,
        available_actions=available_actions,
        initial_state_pgc=initial_state,
        labels=labels,
        rewards=rewards,
    )

    # we build the model in the regular way:
    regular_model = model.new_mdp(create_initial_state=False)
    state1 = regular_model.new_state(labels=["init", "1"])
    state2 = regular_model.new_state(labels=["2"])
    state0 = regular_model.new_state(labels=["0"])
    left = regular_model.new_action(frozenset({"left"}))
    right = regular_model.new_action(frozenset({"right"}))
    branch12 = model.Branch([(0.5, state1), (0.5, state2)])
    branch10 = model.Branch([(0.5, state1), (0.5, state0)])
    branch01 = model.Branch([(0.5, state0), (0.5, state1)])
    branch21 = model.Branch([(0.5, state2), (0.5, state1)])

    regular_model.add_transitions(
        state1, model.Transition({left: branch12, right: branch10})
    )
    regular_model.add_transitions(state2, model.Transition({right: branch21}))
    regular_model.add_transitions(state0, model.Transition({left: branch01}))

    rewardmodel = regular_model.add_rewards("rewardmodel: 0")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.add_rewards("rewardmodel: 1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    assert regular_model == pgc_model


def test_pgc_dtmc():
    # we build the model with pgc:
    p = 0.5
    initial_state = pgc.State(s=0)

    def rewards(s: pgc.State):
        return [1, 2]

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

    pgc_model = pgc.build_pgc(
        delta=delta,
        initial_state_pgc=initial_state,
        rewards=rewards,
        modeltype=model.ModelType.DTMC,
    )

    # we build the model in the regular way:
    regular_model = model.new_dtmc()
    regular_model.states[0].features = {"s": 0}
    regular_model.set_transitions(
        regular_model.get_initial_state(),
        [
            (1 / 2, regular_model.new_state(features={"s": 1})),
            (1 / 2, regular_model.new_state(features={"s": 2})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(1),
        [
            (1 / 2, regular_model.new_state(features={"s": 3})),
            (1 / 2, regular_model.new_state(features={"s": 4})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(2),
        [
            (1 / 2, regular_model.new_state(features={"s": 5})),
            (1 / 2, regular_model.new_state(features={"s": 6})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(3),
        [
            (1 / 2, regular_model.get_state_by_id(1)),
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 1})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(4),
        [
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 2})),
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 3})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(5),
        [
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 4})),
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 5})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(6),
        [
            (1 / 2, regular_model.get_state_by_id(2)),
            (1 / 2, regular_model.new_state(features={"s": 7, "d": 6})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(7),
        [(1, regular_model.new_state(features={"s": 7}))],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(8), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(9), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(10), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(11), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(12), [(1, regular_model.get_state_by_id(13))]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(13), [(1, regular_model.get_state_by_id(13))]
    )

    rewardmodel = regular_model.add_rewards("rewardmodel: 0")
    for state in regular_model.states.values():
        rewardmodel.set_state_reward(state, 1)
    rewardmodel = regular_model.add_rewards("rewardmodel: 1")
    for state in regular_model.states.values():
        rewardmodel.set_state_reward(state, 2)

    assert pgc_model == regular_model


def test_pgc_dtmc_arbitrary():
    def delta(current_state):
        match current_state:
            case "hungry":
                return [(1.0, "eating")]
            case "eating":
                return [(1.0, "hungry")]

    pgc_model = pgc.build_pgc(
        delta, initial_state_pgc="hungry", modeltype=model.ModelType.DTMC
    )

    regular_model = model.new_dtmc()
    regular_model.set_transitions(
        regular_model.get_initial_state(), [(1, regular_model.new_state())]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(1), [(1, regular_model.get_initial_state())]
    )

    assert pgc_model == regular_model
