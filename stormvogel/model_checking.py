import stormpy
import stormvogel.mapping
import stormvogel.result
import stormvogel.model


def model_checking(
    model: stormvogel.model.Model, prop: str, scheduler: bool = True
) -> stormvogel.result.Result | None:
    """
    Instead of calling this function, the stormpy model checker can be used by first mapping a model to a stormpy model,
    then calling the stormpy model checker with it followed by converting the model checker result to a stormvogel result.
    This function just performs this procedure automatically.
    """

    prop = stormpy.parse_properties(prop)

    stormpy_model = stormvogel.mapping.stormvogel_to_stormpy(model)

    if model.supports_actions() and scheduler:
        stormpy_result = stormpy.model_checking(
            stormpy_model, prop[0], extract_scheduler=True
        )
    else:
        stormpy_result = stormpy.model_checking(stormpy_model, prop[0])

    # to get the correct action labels, we need to convert the model back to stormvogel instead of
    # using the initial one for now. (otherwise schedulers won't work)
    stormvogel_model = stormvogel.mapping.stormpy_to_stormvogel(stormpy_model)

    assert stormvogel_model is not None

    stormvogel_result = stormvogel.result.convert_model_checking_result(
        stormvogel_model, stormpy_result
    )

    return stormvogel_result


class PropertyBuilder:
    """
    Aimed to let beginner users build property strings

    Args:
    model: the model that the property string will be used for
    """

    model: stormvogel.model.Model
    prop: str

    def __init__(self, model: stormvogel.model.Model):
        self.model = model
        self.prop = ""

    def get_reachabilty_probability(self, label: str, value: str = "default") -> str:
        """To create a property that asks for the reachability probabilities of the states"""

        print(value)

        assert len(self.model.get_states_with_label(label)) > 0

        if value == "default":
            string = f'P=? [F "{label}"]'
            self.prop = string
            return string
        elif value == "max":
            string = f'Pmax=? [F "{label}"]'
            self.prop = string
            return string
        elif value == "min":
            string = f'Pmin=? [F "{label}"]'
            self.prop = string
            return string
        else:
            raise RuntimeError(
                "Please choose as value either 'max', 'min' or 'default'."
            )
