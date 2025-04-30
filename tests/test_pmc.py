import stormvogel.examples.pmc
import stormvogel.stormpy_utils.mapping as mapping


def test_pmc():
    dtmc = stormvogel.examples.pmc.create_die_dtmc()

    # we test the mapping
    stormpy_dtmc = mapping.stormvogel_to_stormpy(dtmc)
    new_dtmc = mapping.stormpy_to_stormvogel(stormpy_dtmc)
    assert dtmc == new_dtmc

    # TODO valuations



    print(dtmc)
