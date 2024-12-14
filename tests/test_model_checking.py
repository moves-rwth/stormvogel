import stormvogel.model_checking
import examples.monty_hall
import stormpy


def test_mdp_model_checking():
    # we get our result using the stormvogel model checker function indirectly
    mdp = examples.monty_hall.create_monty_hall_mdp()
    prop = 'Pmin=? [F "done"]'
    result = stormvogel.model_checking.model_checking(mdp, prop, True)

    # and directly
    prop = stormpy.parse_properties(prop)

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(mdp)
    stormpy_result = stormpy.model_checking(
        stormpy_model, prop[0], extract_scheduler=True
    )
    assert mdp is not None

    # to get the correct action labels, we need to convert the model back to stormvogel instead of
    # using the initial one for now. (otherwise schedulers won't work)
    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(stormpy_model)

    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    assert result == stormvogel_result
