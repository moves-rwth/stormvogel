import stormvogel.model
import examples.monty_hall
import pytest


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
    probabilities = [prob for prob, state in transitions]

    # TODO also compare states (needs simpler representation of transition tuples)
    assert len(transitions) == 3
    assert pytest.approx(probabilities) == [1 / 3, 1 / 3, 1 / 3]


"""
def test_transition_from_shorthand():


def test_is_well_defined():



def test_normalize():



def test_delete_state():



def test_delete_transitions_between_states()
"""
