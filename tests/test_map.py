import examples.monty_hall
import examples.stormpy_mdp
import stormvogel.map
import stormvogel.model
import examples.stormpy_dtmc
import examples.die
import examples.stormpy_ctmc
import examples.stormpy_pomdp
import examples.nuclear_fusion_ctmc
import examples.monty_hall_pomdp
import examples.stormpy_ma
import stormpy


def matrix_equals(
    model0: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    model1: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
) -> bool:
    """outputs true if the sparsematrices of the two sparsedtmcs are the same and false otherwise"""

    # TODO is there a better check for equality for matrices in storm(py)? otherwise one should perhaps be implemented
    return str(model0.transition_matrix) == str(model1.transition_matrix)


def test_stormpy_to_stormvogel_and_back_dtmc():
    # we test it for an example stormpy representation of a dtmc
    stormpy_dtmc = examples.stormpy_dtmc.example_building_dtmcs_01()
    # print(stormpy_dtmc.transition_matrix)
    stormvogel_dtmc = stormvogel.map.stormpy_to_stormvogel(stormpy_dtmc)
    # print(stormvogel_dtmc)
    assert stormvogel_dtmc is not None
    new_stormpy_dtmc = stormvogel.map.stormvogel_to_stormpy(stormvogel_dtmc)
    # print(new_stormpy_dtmc.transition_matrix)

    # TODO also compare other parts than the matrix (e.g. state labels)
    assert matrix_equals(stormpy_dtmc, new_stormpy_dtmc)


def test_stormvogel_to_stormpy_and_back_dtmc():
    # we test it for the die dtmc
    stormvogel_dtmc = examples.die.create_die_dtmc()
    # print(stormvogel_dtmc)
    stormpy_dtmc = stormvogel.map.stormvogel_to_stormpy(stormvogel_dtmc)
    # print(stormpy_dtmc)
    new_stormvogel_dtmc = stormvogel.map.stormpy_to_stormvogel(stormpy_dtmc)
    # print(new_stormvogel_dtmc)

    assert new_stormvogel_dtmc == stormvogel_dtmc


def test_stormpy_to_stormvogel_and_back_mdp():
    # we test it for an example stormpy representation of an mdp
    stormpy_mdp = examples.stormpy_mdp.example_building_mdps_01()
    # print(stormpy_mdp)
    stormvogel_mdp = stormvogel.map.stormpy_to_stormvogel(stormpy_mdp)
    # print(stormvogel_mdp)
    assert stormvogel_mdp is not None
    new_stormpy_mdp = stormvogel.map.stormvogel_to_stormpy(stormvogel_mdp)
    # print(new_stormpy_mdp)

    # TODO also compare other parts than the matrix (e.g. choice labels)
    assert matrix_equals(stormpy_mdp, new_stormpy_mdp)


def test_stormvogel_to_stormpy_and_back_mdp():
    # we test it for monty hall mdp
    stormvogel_mdp = examples.monty_hall.create_monty_hall_mdp()
    # print(stormvogel_mdp)
    stormpy_mdp = stormvogel.map.stormvogel_to_stormpy(stormvogel_mdp)
    # print(stormpy_mdp)
    new_stormvogel_mdp = stormvogel.map.stormpy_to_stormvogel(stormpy_mdp)
    # print(new_stormvogel_mdp)

    assert new_stormvogel_mdp == stormvogel_mdp


def test_stormvogel_to_stormpy_and_back_ctmc():
    # we create a stormpy representation of an example ctmc
    stormvogel_ctmc = examples.nuclear_fusion_ctmc.create_nuclear_fusion_ctmc()
    # print(stormvogel_ctmc)
    stormpy_ctmc = stormvogel.map.stormvogel_to_stormpy(stormvogel_ctmc)
    # print(stormpy_ctmc)
    new_stormvogel_ctmc = stormvogel.map.stormpy_to_stormvogel(stormpy_ctmc)
    # print(new_stormvogel_ctmc)

    assert new_stormvogel_ctmc == stormvogel_ctmc


def test_stormpy_to_stormvogel_and_back_ctmc():
    # we create a stormpy representation of an example ctmc
    stormpy_ctmc = examples.stormpy_ctmc.example_building_ctmcs_01()
    # print(stormpy_ctmc)
    stormvogel_ctmc = stormvogel.map.stormpy_to_stormvogel(stormpy_ctmc)
    # print(stormvogel_ctmc)
    assert stormvogel_ctmc is not None
    new_stormpy_ctmc = stormvogel.map.stormvogel_to_stormpy(stormvogel_ctmc)
    # print(new_stormpy_ctmc)

    assert matrix_equals(stormpy_ctmc, new_stormpy_ctmc)


def test_stormvogel_to_stormpy_and_back_pomdp():
    # we create a stormpy representation of an example pomdp
    stormvogel_pomdp = examples.monty_hall_pomdp.create_monty_hall_pomdp()
    # print(stormvogel_pomdp)
    stormpy_pomdp = stormvogel.map.stormvogel_to_stormpy(stormvogel_pomdp)
    # print(stormpy_pomdp)
    new_stormvogel_pomdp = stormvogel.map.stormpy_to_stormvogel(stormpy_pomdp)
    # print(new_stormvogel_pomdp)

    assert new_stormvogel_pomdp == stormvogel_pomdp


def test_stormpy_to_stormvogel_and_back_pomdp():
    # we create a stormpy representation of an example pomdp
    stormpy_pomdp = examples.stormpy_pomdp.example_building_pomdps_01()
    # print(stormpy_pomdp)
    stormvogel_pomdp = stormvogel.map.stormpy_to_stormvogel(stormpy_pomdp)
    # print(stormvogel_pomdp)
    assert stormvogel_pomdp is not None
    new_stormpy_pomdp = stormvogel.map.stormvogel_to_stormpy(stormvogel_pomdp)
    # print(new_stormpy_pomdp)

    assert matrix_equals(stormpy_pomdp, new_stormpy_pomdp)


"""
def test_stormvogel_to_stormpy_and_back_ma():
    # we create a stormpy representation of an example ma
    stormvogel_ma = examples.monty_hall_ma.create_monty_hall_ma()
    # print(stormvogel_ma)
    stormpy_ma = stormvogel.map.stormvogel_to_stormpy(stormvogel_ma)
    # print(stormpy_ma)
    new_stormvogel_ma = stormvogel.map.stormpy_to_stormvogel(stormpy_ma)
    # print(new_stormvogel_ma)

    assert new_stormvogel_ma == stormvogel_ma
"""


def test_stormpy_to_stormvogel_and_back_ma():
    # we create a stormpy representation of an example ma
    stormpy_ma = examples.stormpy_ma.example_building_mas_01()
    # print(stormpy_ma)
    stormvogel_ma = stormvogel.map.stormpy_to_stormvogel(stormpy_ma)
    # print(stormvogel_ma)
    assert stormvogel_ma is not None
    new_stormpy_ma = stormvogel.map.stormvogel_to_stormpy(stormvogel_ma)
    # print(new_stormpy_ma)

    assert matrix_equals(stormpy_ma, new_stormpy_ma)
