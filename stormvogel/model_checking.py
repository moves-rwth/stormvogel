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
    stormvogel_result = stormvogel.result.convert_model_checking_result(
        model, stormpy_result
    )

    return stormvogel_result


if __name__ == "__main__":
    dtmc = stormvogel.model.new_dtmc("Die")
    init = dtmc.get_initial_state()
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )
    dtmc.add_self_loops()

    stormvogel_results = model_checking(dtmc, 'Pmin=? [F "rolled1"]')
    print(stormvogel_results)
