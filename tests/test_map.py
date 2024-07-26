import examples.monty_hall
import examples.stormpy_mdp
import stormvogel.map
import stormvogel.model
import examples.stormpy_to_stormvogel
import examples.die
import stormpy


def matrix_equals(
    model0: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    model1: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
) -> bool:
    # outputs true if the sparsematrices of the two sparsedtmcs are the same and false otherwise

    # TODO is there a better check for equality for matrices in storm(py)? otherwise one should perhaps be implemented
    return str(model0.transition_matrix) == str(model1.transition_matrix)


def test_stormpy_to_stormvogel_and_back_dtmc():
    stormpy_dtmc = examples.stormpy_to_stormvogel.example_building_dtmcs_01()
    # print(stormpy_dtmc.transition_matrix)
    stormvogel_dtmc = stormvogel.map.stormpy_to_stormvogel(stormpy_dtmc)
    # print(stormvogel_dtmc)
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
    # we test it for monty hall mdp
    stormpy_mdp = examples.stormpy_mdp.example_building_mdps_01()
    # print(stormpy_mdp)
    stormvogel_mdp = stormvogel.map.stormpy_to_stormvogel(stormpy_mdp)
    # print(stormvogel_mdp)
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
