import stormpy
import stormvogel.model
import stormpy.examples.files
import stormpy.examples


class Result:
    """Result object represents the model checking results for a given model

    Args:
        probabilities: for each state the model checking result
        scheduler: in case the model is an mdp we can optionally store a scheduler
    """

    probabilities: list[int]
    scheduler: dict[int, stormvogel.model.Action] | None

    def __init__(self, probabilities: list[int]):
        self.probabilities = probabilities
        self.scheduler = None

    def add_scheduler(self, scheduler: dict[int, stormvogel.model.Action] | None):
        """adds a scheduler to the results"""
        self.scheduler = scheduler

    def get_result_of_state(self, state: stormvogel.model.State):
        """returns the model checking result for a given state"""
        return self.probabilities[state.id]


def convert_model_checking_results(
    stormpy_result: stormpy.core.ExplicitQuantitativeCheckResult,
) -> Result:
    """
    Takes a model checking result from stormpy and converts it to a stormvogel representation
    """
    # print(dir(result))

    # print(result.get_values())

    stormvogel_result = Result(stormpy_result.get_values())
    if stormpy_result.has_scheduler:
        print("scheduler:", stormpy_result.scheduler)
        scheduler = None
        stormvogel_result.add_scheduler(scheduler)

    return stormvogel_result


if __name__ == "__main__":
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])
    # print(type(result))
    convert_model_checking_results(result)
