import examples.die
import stormvogel.property_builder


def test_property_string_builder():
    # we want to create this string 'Pmin=? [F "rolled1"]':

    dtmc = examples.die.create_die_dtmc()
    prop_builder = stormvogel.model_checking.PropertyBuilder(dtmc)
    prop = prop_builder.get_reachabilty_probability("rolled1")

    assert prop == 'P=? [F "rolled1"]'
