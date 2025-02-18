import stormvogel.model_checking
import examples.monty_hall
import examples.die


try:
    import stormpy

    stormpy_installed = True
except ImportError:
    stormpy_installed = False


def test_model_checking():
    # TODO this test is maybe too trivial?

    # we get our result using the stormvogel model checker function indirectly
    mdp = examples.monty_hall.create_monty_hall_mdp()
    prop = 'Pmax=? [F "done"]'
    result = stormvogel.model_checking.model_checking(mdp, prop, True)

    # and directly
    prop = stormpy.parse_properties(prop)

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(mdp)
    stormpy_result = stormpy.model_checking(
        stormpy_model, prop[0], extract_scheduler=True
    )

    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(stormpy_model)

    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    # now we do it for a dtmc:
    dtmc = examples.die.create_die_dtmc()
    prop = 'P=? [F "rolled1"]'
    result = stormvogel.model_checking.model_checking(dtmc, prop, True)

    # indirectly:
    prop = stormpy.parse_properties(prop)
    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(dtmc)
    stormpy_result = stormpy.model_checking(stormpy_model, prop[0])

    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(stormpy_model)

    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    assert result == stormvogel_result
