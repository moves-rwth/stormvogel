import stormvogel.examples.pmc
import stormvogel.stormpy_utils.mapping as mapping


def test_pmc_conversion():
    pmc = stormvogel.examples.pmc.create_simple_pmc()

    # we test the mapping
    stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
    new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)
    assert pmc == new_pmc


# def test_pmdp_conversion():
#    pmdp = stormvogel.examples.pmc.create_simple_pmdp()
#
#    # we test the mapping
#    stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)
#    new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)
#    assert pmdp == new_pmdp


# def test_pmc_valuations():
#    pmc = stormvogel.examples.pmc.create_simple_pmc()
#    # pmc.valuate()
#    # TODO valuations
