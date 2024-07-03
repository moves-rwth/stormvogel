# content of test_sysexit.py
import stormvogel.model
from stormvogel.model import EmptyAction


def test_mdp_creation():
    dtmc = stormvogel.model.new_dtmc("Die")

    init = dtmc.get_initial_state()

    # roll die
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    assert len(dtmc.states) == 7
    assert len(dtmc.transitions) == 7
    # Check that all states 1..6 have self loops
    for i in range(1, 7):
        # yeah we need transition getting syntax
        assert dtmc.transitions[i].transition[EmptyAction].branch[0][1].id == i
