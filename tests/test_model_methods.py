import stormvogel.model
import examples.monty_hall
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


def test_is_well_defined():
    # we check for an instance where it is not well defined
    dtmc = stormvogel.model.new_dtmc()
    state = dtmc.new_state()
    dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1 / 2, state)],
    )

    assert not dtmc.is_well_defined()

    # we check for an instance where it is well defined
    dtmc.set_transitions(
        dtmc.get_initial_state(),
        [(1 / 2, state), (1 / 2, state)],
    )

    dtmc.add_self_loops()

    assert dtmc.is_well_defined()


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
    ctmc.remove_state(ctmc.get_state_by_id(3), True)

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
