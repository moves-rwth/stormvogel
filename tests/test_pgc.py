from stormvogel import pgc, model
import math
import pytest
import re


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
        return {"r1": 1, "r2": 2}

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

    rewardmodel = regular_model.add_rewards("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.add_rewards("r2")
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
        return {"r1": 1, "r2": 2}

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
        state1,
        model.Transition(
            {right: branch10, left: branch12}
        ),  # state1, model.Transition({left: branch12, right: branch10})
    )
    regular_model.add_transitions(state2, model.Transition({right: branch21}))
    regular_model.add_transitions(state0, model.Transition({left: branch01}))

    rewardmodel = regular_model.add_rewards("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.add_rewards("r2")
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
        return {"r1": 1, "r2": 2}

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
    regular_model.states[0].valuations = {"s": 0}
    regular_model.set_transitions(
        regular_model.get_initial_state(),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 1})),
            (1 / 2, regular_model.new_state(valuationss={"s": 2})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(1),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 3})),
            (1 / 2, regular_model.new_state(valuations={"s": 4})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(2),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 5})),
            (1 / 2, regular_model.new_state(valuations={"s": 6})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(3),
        [
            (1 / 2, regular_model.get_state_by_id(1)),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 1})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(4),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 2})),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 3})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(5),
        [
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 4})),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 5})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(6),
        [
            (1 / 2, regular_model.get_state_by_id(2)),
            (1 / 2, regular_model.new_state(valuations={"s": 7, "d": 6})),
        ],
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(7),
        [(1, regular_model.new_state(valuations={"s": 7}))],
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

    rewardmodel = regular_model.add_rewards("r1")
    for state in regular_model.states.values():
        rewardmodel.set_state_reward(state, 1)
    rewardmodel = regular_model.add_rewards("r2")
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


def test_pgc_mdp_empty_action():
    # we test if we can also provide empty actions
    def available_actions(s):
        return [pgc.Action([])]

    def delta(current_state, action):
        match current_state:
            case "hungry":
                return [(1.0, "eating")]
            case "eating":
                return [(1.0, "hungry")]

    pgc_model = pgc.build_pgc(
        delta,
        initial_state_pgc="hungry",
        available_actions=available_actions,
        modeltype=model.ModelType.MDP,
    )

    regular_model = model.new_mdp()
    regular_model.set_transitions(
        regular_model.get_initial_state(), [(1, regular_model.new_state())]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(1), [(1, regular_model.get_initial_state())]
    )

    assert pgc_model == regular_model


def test_pgc_endless():
    init = pgc.State(x="")

    def available_actions(s: pgc.State):
        if s == init:  # If we are in the initial state, we have a choice.
            return [pgc.Action(["study"]), pgc.Action(["don't study"])]
        else:  # Otherwise, we don't have any choice, we are just a Markov chain.
            return [pgc.Action([])]

    def delta(s: pgc.State, a: pgc.Action):
        if "study" in a.labels:
            return [(1, pgc.State(x=["studied"]))]
        elif "don't study" in a.labels:
            return [(1, pgc.State(x=["didn't study"]))]
        elif "studied" in s.x:
            return [
                (9 / 10, pgc.State(x=["pass test"])),
                (1 / 10, pgc.State(x=["fail test"])),
            ]
        elif "didn't study" in s.x:
            return [
                (2 / 5, pgc.State(x=["pass test"])),
                (3 / 5, pgc.State(x=["fail test"])),
            ]
        else:
            return [(1, pgc.State(x=[f"{s.x[0]}0"]))]

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "The model you want te create has a very large amount of states (at least 2000), if you wish to proceed, set max_size to some larger number."
        ),
    ):
        pgc.build_pgc(
            delta=delta,
            initial_state_pgc=init,
            available_actions=available_actions,
            modeltype=model.ModelType.MDP,
        )


def test_pgc_pomdp():
    # here we test if the observations function works
    # we build the pomdp model with pgc:
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
        return {"r1": 1, "r2": 2}

    def labels(s: pgc.State):
        return [str(s.x)]

    def observations(s: pgc.State):
        return 5

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
        observations=observations,
        modeltype=model.ModelType.POMDP,
    )

    # we build the pomdp model in the regular way:
    regular_model = model.new_pomdp(create_initial_state=False)
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

    rewardmodel = regular_model.add_rewards("r1")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 1)
    rewardmodel = regular_model.add_rewards("r2")
    for i in range(2 * N):
        pair = regular_model.get_state_action_pair(i)
        assert pair is not None
        rewardmodel.set_state_action_reward(pair[0], pair[1], 2)

    for state in regular_model.states.values():
        state.set_observation(5)

    assert regular_model == pgc_model


def test_pgc_ctmc():
    def delta(current_state):
        match current_state:
            case "hungry":
                return [(5.0, "eating")]
            case "eating":
                return [(3.0, "hungry")]

    def rates(s):
        match s:
            case "hungry":
                return 5
            case "eating":
                return 3

    pgc_model = pgc.build_pgc(
        delta, initial_state_pgc="hungry", rates=rates, modeltype=model.ModelType.CTMC
    )

    regular_model = model.new_ctmc()
    regular_model.set_transitions(
        regular_model.get_initial_state(), [(5, regular_model.new_state())]
    )
    regular_model.set_transitions(
        regular_model.get_state_by_id(1), [(3, regular_model.get_initial_state())]
    )
    regular_model.set_rate(regular_model.get_initial_state(), 5)
    regular_model.set_rate(list(regular_model.states.values())[1], 3)

    assert pgc_model == regular_model
