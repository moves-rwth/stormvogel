import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.stormpy_utils.convert_results as convert_results
import stormvogel.model
import stormvogel.property_builder

try:
    import stormpy
except ImportError:
    stormpy = None


def model_checking(
    model: stormvogel.model.Model, prop: str | None = None, scheduler: bool = True
) -> stormvogel.result.Result | None:
    """
    Instead of calling this function, the stormpy model checker can be used by first mapping a model to a stormpy model,
    then calling the stormpy model checker with it followed by converting the model checker result to a stormvogel result.
    This function just performs this procedure automatically.
    """

    assert stormpy is not None
    if prop:
        prop = stormpy.parse_properties(prop)

        stormpy_model = mapping.stormvogel_to_stormpy(model)

        assert prop is not None
        if model.supports_actions() and scheduler:
            stormpy_result = stormpy.model_checking(
                stormpy_model, prop[0], extract_scheduler=True
            )
        else:
            stormpy_result = stormpy.model_checking(stormpy_model, prop[0])

        # to get the correct action labels, we need to convert the model back to stormvogel instead of
        # using the initial one for now. (otherwise schedulers won't work)
        stormvogel_model = mapping.stormpy_to_stormvogel(stormpy_model)

        assert stormvogel_model is not None

        stormvogel_result = convert_results.convert_model_checking_result(
            stormvogel_model, stormpy_result
        )

        return stormvogel_result
    else:
        print(
            "You have not proved a property string. You can create a simple one using this widget."
        )
        stormvogel.property_builder.build_property_string(model)
        return None


if __name__ == "__main__":
    import examples.monty_hall

    mdp = examples.monty_hall.create_monty_hall_mdp()

    rewardmodel = mdp.add_rewards("rewardmodel")
    rewardmodel.set_from_rewards_vector(list(range(67)))
    rewardmodel2 = mdp.add_rewards("rewardmodel2")
    rewardmodel2.set_from_rewards_vector(list(range(67)))

    print(model_checking(mdp))  # ,'R{"rewardmodel"}min=? [F "target" | "done"]'))
