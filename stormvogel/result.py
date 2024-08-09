import stormpy
import stormvogel.model
import stormpy.examples.files
import stormpy.examples


class Result:
    """Result object represents the model checking results for a given model

    Args:
        values: for each state the model checking result
        scheduler: in case the model is an mdp we can optionally store a scheduler
    """

    values: list[stormvogel.model.Number]

    # TODO link states to actual actions instead of only a string?
    scheduler: dict[int, str] | None

    def __init__(self, values: list[stormvogel.model.Number]):
        self.values = values
        self.scheduler = None

    def add_scheduler(self, stormpy_scheduler: stormpy.storage.Scheduler):
        """adds a scheduler to the result"""
        stormvogel_scheduler = {}
        for i in range(len(self.values)):
            stormvogel_scheduler[i] = str(stormpy_scheduler.get_choice(i))
        self.scheduler = stormvogel_scheduler

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Number:
        """returns the model checking result for a given state"""
        return self.values[state.id]

    def get_choice_of_state(self, state: stormvogel.model.State) -> str | None:
        """returns the choice in the sceduler for the given state"""
        if self.scheduler is not None:
            return self.scheduler[state.id]
        else:
            print("this result does not have a scheduler")
            return None


def convert_model_checking_result(
    stormpy_result: stormpy.core.ExplicitQuantitativeCheckResult,
    with_scheduler: bool = True,
) -> Result:
    """
    Takes a model checking result from stormpy and converts it to a stormvogel representation
    """
    stormvogel_result = Result(stormpy_result.get_values())
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
    convert_model_checking_result(result)
