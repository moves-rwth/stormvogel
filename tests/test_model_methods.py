import stormvogel.model
import examples.monty_hall
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


"""
def test_normalize():


def test_delete_state():

def test_delete_transitions_between_states()
"""
