import stormpy
import stormvogel.mapping
import stormvogel.result
import stormvogel.model


def model_checking(
    model: stormvogel.model.Model, prop: str, scheduler: bool = False
) -> stormvogel.result.Result | None:
    """
    Instead of calling this function, the stormpy model checker can be used by first mapping a model to a stormpy model,
    then calling the stormpy model checker with it followed by converting the model checker result to a stormvogel result.
    This function just performs this procedure automatically.
    """

    prop = stormpy.parse_properties(prop)

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    stormpy_result = stormpy.model_checking(
        stormpy_model, prop[0], extract_scheduler=scheduler
    )
    assert model is not None

    # to get the correct action labels, we need to convert the model back to stormvogel instead of
    # using the initial one for now. (otherwise schedulers won't work)
    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(stormpy_model)

    assert stormvogel_model is not None

    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    return stormvogel_result
