import stormvogel.model
import examples.monty_hall
import examples.die
import examples.nuclear_fusion_ctmc
import pytest
from typing import cast


def test_available_actions():
    mdp = examples.monty_hall.create_monty_hall_mdp()

    action = [
        stormvogel.model.Action(name="open0", labels=frozenset()),
        stormvogel.model.Action(name="open1", labels=frozenset()),
        stormvogel.model.Action(name="open2", labels=frozenset()),
    ]
    assert mdp.get_state_by_id(1).available_actions() == action


def test_get_outgoing_transitions():
    mdp = examples.monty_hall.create_monty_hall_mdp()

    transitions = mdp.get_initial_state().get_outgoing_transitions(
        stormvogel.model.Action(name="empty", labels=frozenset())
    )

    probabilities, states = zip(*transitions)

    assert pytest.approx(list(probabilities)) == [1 / 3, 1 / 3, 1 / 3]
    assert list(states) == [
        mdp.get_state_by_id(1),
        mdp.get_state_by_id(2),
        mdp.get_state_by_id(3),
    ]


def test_is_absorbing():
    # one example of a ctmc state that is absorbing and one that isn't
    ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    state0 = ctmc.get_state_by_id(4)
    state1 = ctmc.get_state_by_id(3)
    assert state0.is_absorbing()
    assert not state1.is_absorbing()

    # one example of a dtmc state that is absorbing and one that isn't
    dtmc = examples.die.create_die_dtmc()
    state0 = dtmc.get_initial_state()
    state1 = dtmc.get_state_by_id(1)
    assert state1.is_absorbing()
    assert not state0.is_absorbing()


def test_transition_from_shorthand():
    # First we test it for a model without actions
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    transition_shorthand = [(1 / 2, state)]
    branch = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Number, stormvogel.model.State]],
            transition_shorthand,
        )
    )
    action = stormvogel.model.EmptyAction
    transition = stormvogel.model.Transition({action: branch})

    assert (
        stormvogel.model.transition_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Number, stormvogel.model.State]],
                transition_shorthand,
            )
        )
        == transition
    )

    # Then we test it for a model with actions
    mdp = stormvogel.model.new_mdp()
    state = mdp.new_state()
    action = mdp.new_action("0", frozenset("action"))
    transition_shorthand = [(action, state)]
    branch = stormvogel.model.Branch(
        cast(list[tuple[stormvogel.model.Number, stormvogel.model.State]], [(1, state)])
    )
    transition = stormvogel.model.Transition({action: branch})

    assert (
        stormvogel.model.transition_from_shorthand(
            cast(
                list[tuple[stormvogel.model.Action, stormvogel.model.State]],
                transition_shorthand,
            )
        )
        == transition
    )


def test_is_stochastic():
    # we check for an instance where it is not stochastic
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1 / 2, state)],
    )

    assert not dtmc.is_stochastic()

    # we check for an instance where it is stochastic
    dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1 / 2, state), (1 / 2, state)],
    )

    dtmc.add_self_loops()

    assert dtmc.is_stochastic()

    # we check it for a continuous time model
    ctmc = stormvogel.model.new_ctmc()
    ctmc.set_transitions(ctmc.get_initial_state(), [(1, ctmc.new_state())])

    ctmc.add_self_loops()

    assert not ctmc.is_stochastic()


def test_normalize():
    # we make a dtmc that has outgoing transitions with sum of probabilities != 0 and we normalize it
    dtmc0 = stormvogel.model.new_dtmc()
    state = dtmc0.new_state()
    dtmc0.set_transitions(
        dtmc0.get_initial_state(),
        [(1 / 4, state), (1 / 2, state)],
    )
    dtmc0.add_self_loops()
    dtmc0.normalize()

    # we make the same dtmc but with the already normalized probabilities
    dtmc1 = stormvogel.model.new_dtmc()
    state = dtmc1.new_state()
    dtmc1.set_transitions(
        dtmc1.get_initial_state(),
        [(1 / 3, state), (2 / 3, state)],
    )
    dtmc1.add_self_loops()

    assert dtmc0 == dtmc1


def test_remove_state():
    # we make a normal ctmc and remove a state
    ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    ctmc.remove_state(ctmc.get_state_by_id(3))

    # we make a ctmc with the state already missing
    new_ctmc = stormvogel.model.new_ctmc("Nuclear fusion")
    new_ctmc.get_state_by_id(0).set_transitions([(3, new_ctmc.new_state("helium"))])
    new_ctmc.get_state_by_id(1).set_transitions([(2, new_ctmc.new_state("carbon"))])

    new_ctmc.new_state("Supernova")

    rates = [3, 2, 7, 0]
    for i in range(4):
        new_ctmc.set_rate(new_ctmc.get_state_by_id(i), rates[i])
    new_ctmc.add_self_loops()

    assert ctmc == new_ctmc

    # we also test if it works for a model that has nontrivial actions:
    mdp = stormvogel.model.new_mdp()
    state1 = mdp.new_state()
    state2 = mdp.new_state()
    action0 = mdp.new_action("0")
    action1 = mdp.new_action("1")
    branch0 = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Number, stormvogel.model.State]],
            [(1 / 2, state1), (1 / 2, state2)],
        )
    )
    branch1 = stormvogel.model.Branch(
        cast(
            list[tuple[stormvogel.model.Number, stormvogel.model.State]],
            [(1 / 4, state1), (3 / 4, state2)],
        )
    )
    transition = stormvogel.model.Transition({action0: branch0, action1: branch1})
    mdp.set_transitions(mdp.get_initial_state(), transition)

    # we remove a state
    mdp.remove_state(mdp.get_state_by_id(0))

    # we make the mdp with the state already missing
    new_mdp = stormvogel.model.new_mdp(create_initial_state=False)
    new_mdp.new_state()
    new_mdp.new_state()
    new_mdp.add_self_loops()

    assert mdp == new_mdp


def test_remove_transitions_between_states():
    # we make a model and remove transitions between two states
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1, state)],
    )
    dtmc.set_transitions(state, [(1, dtmc.get_initial_state())])
    dtmc.remove_transitions_between_states(state, dtmc.get_initial_state())

    # we create a model with the transitions between the two states already missing
    new_dtmc = stormvogel.model.new_dtmc()
    state = new_dtmc.new_state()
    new_dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1, state)],
    )
    new_dtmc.add_self_loops()

    assert dtmc == new_dtmc


def test_add_transitions():
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    # A non-action model should throw an exception.
    # with pytest.raises(RuntimeError) as excinfo:
    #    dtmc.add_transitions(
    #        dtmc.get_initial_state(),
    #        [(0.5, state)],
    #    )
    # assert (
    #    str(excinfo.value)
    #    == "Models without actions do not support add_transitions. Use set_transitions instead."
    # )

    # Empty transition case, act exactly like set_transitions.
    mdp = stormvogel.model.new_mdp()
    state = mdp.new_state()
    mdp.add_transitions(
        mdp.get_initial_state(),
        [(0.5, state)],
    )
    mdp2 = stormvogel.model.new_mdp()
    state2 = mdp2.new_state()
    mdp2.set_transitions(
        mdp2.get_initial_state(),
        [(0.5, state2)],
    )
    assert mdp == mdp2

    # Fail to add a real action to an empty action.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    mdp3.set_transitions(
        mdp3.get_initial_state(),
        [(0.5, state3)],
    )
    action3 = mdp3.new_action("action")
    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_transitions(mdp3.get_initial_state(), [(action3, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a transition with an non-empty action to a transition which has an empty action. Use set_transition instead."
    )
    # And the other way round.
    mdp3 = stormvogel.model.new_mdp()
    state3 = mdp2.new_state()
    action3 = mdp3.new_action("action")
    mdp3.set_transitions(
        mdp3.get_initial_state(),
        [(action3, state3)],
    )

    with pytest.raises(RuntimeError) as excinfo:
        mdp3.add_transitions(mdp3.get_initial_state(), [(0.5, state3)])
    assert (
        str(excinfo.value)
        == "You cannot add a transition with an empty action to a transition which has no empty action. Use set_transition instead."
    )

    # Empty action case, add the branches together.
    mdp5 = stormvogel.model.new_mdp()
    state5 = mdp5.new_state()
    mdp5.set_transitions(mdp5.get_initial_state(), [((0.4), state5)])
    mdp5.add_transitions(mdp5.get_initial_state(), [(0.6, state5)])
    assert mdp5.get_branch(mdp5.get_initial_state()).branch == [
        ((0.4), state5),
        (0.6, state5),
    ]

    # Non-empty action case, add the actions to the list.
    mdp6 = stormvogel.model.new_mdp()
    state6 = mdp6.new_state()
    action6a = mdp6.new_action("a")
    action6b = mdp6.new_action("b")
    mdp6.set_transitions(mdp6.get_initial_state(), [(action6a, state6)])
    mdp6.add_transitions(mdp6.get_initial_state(), [(action6b, state6)])
    # print(mdp6.get_transitions(mdp6.get_initial_state()).transition)
    # print([(action6a, state6), (action6b, state6)])
    assert len(mdp6.get_transitions(mdp6.get_initial_state()).transition) == 2
