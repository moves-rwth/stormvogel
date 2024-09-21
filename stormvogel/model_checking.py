import stormpy
import stormvogel.mapping
import stormvogel.result
import stormvogel.model


def model_checking(
    model: stormvogel.model.Model, prop: str | stormpy.Property, scheduler: bool = False
) -> stormvogel.result.Result | None:
    """
    Instead of calling this function, the stormpy model checker can be used by first mapping a model to a stormpy model,
    then calling the stormpy model checker with it followed by converting the model checker result to a stormvogel result.
    This function just performs this procedure automatically.
    """

    # TODO parse properties

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)
    # parsed_prop = stormpy.parse_properties_without_context(prop)
    stormpy_result = stormpy.model_checking(
        stormpy_model, prop, extract_scheduler=scheduler
    )
    assert stormvogel_model is not None
    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    return stormvogel_result


if __name__ == "__main__":
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model(prism_program, properties)

    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(model)

    assert stormvogel_model is not None
    stormvogel_results = model_checking(stormvogel_model, properties[0])
    print(stormvogel_results)
