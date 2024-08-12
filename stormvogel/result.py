import stormpy
import stormvogel.map
import stormvogel.model
import stormpy.examples.files
import stormpy.examples


class Result:
    """Result object represents the model checking results for a given model

    Args:
        model: stormvogel representation of the model associated with the results
        values: for each state the model checking result
        scheduler: in case the model is an mdp we can optionally store a scheduler
    """

    model: stormvogel.model.Model
    values: dict[int, stormvogel.model.Number]
    scheduler: dict[int, stormvogel.model.Action] | None

    def __init__(
        self, model: stormvogel.model.Model, values: list[stormvogel.model.Number]
    ):
        self.model = model

        self.values = {}
        for index, val in enumerate(values):
            self.values[index] = val

        self.scheduler = None

    def add_scheduler(self, stormpy_scheduler: stormpy.storage.Scheduler):
        """adds a scheduler to the result"""
        stormvogel_scheduler = {}
        for i in range(len(self.values)):
            stormvogel_scheduler[i] = self.model.get_action(
                str(stormpy_scheduler.get_choice(i))
            )
        self.scheduler = stormvogel_scheduler

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Number | None:
        """returns the model checking result for a given state if present in the model"""
        if state in self.model.states.values():
            return self.values[state.id]
        else:
            print("This state is not a part of the model")
            return None

    def get_choice_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Action | None:
        """returns the choice in the sceduler for the given state if present in the model"""
        if self.scheduler is not None:
            if state in self.model.states.values():
                return self.scheduler[state.id]
            else:
                print("This state is not a part of the model")
            return None
        else:
            print("this result does not have a scheduler")
            return None


def convert_model_checking_result(
    model: stormvogel.model.Model,
    stormpy_result: stormpy.core.ExplicitQuantitativeCheckResult,
    with_scheduler: bool = True,
) -> Result:
    """
    Takes a model checking result from stormpy and its associated model and converts it to a stormvogel representation
    """
    stormvogel_result = Result(model, stormpy_result.get_values())
    if stormpy_result.has_scheduler and with_scheduler:
        stormvogel_result.add_scheduler(stormpy_result.scheduler)

    return stormvogel_result


if __name__ == "__main__":
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])
    print(type(result))
    stormvogel_model = stormvogel.map.stormpy_to_stormvogel(model)
    if stormvogel_model is not None:
        convert_model_checking_result(stormvogel_model, result)
