import stormpy
import stormvogel.map
import stormvogel.model
import stormpy.examples.files
import stormpy.examples


class Scheduler:
    """
    Scheduler object specifiec what action to take in each state
    """

    model: stormvogel.model.Model
    # taken actions are hashed by the state id
    taken_actions: dict[int, stormvogel.model.Action]

    def __init__(
        self,
        model: stormvogel.model.Model,
        taken_actions: dict[int, stormvogel.model.Action],
    ):
        self.model = model
        self.taken_actions = taken_actions

    def get_choice_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Action | None:
        """returns the choice in the scheduler for the given state if present in the model"""
        if state in self.model.states.values():
            return self.taken_actions[state.id]
        else:
            print("This state is not a part of the model")
            return None

    def __str__(self) -> str:
        if self.model.name is not None:
            add = "Scheduler for model: " + self.model.name + "\n"
        else:
            add = ""
        return add + "taken actions: " + str(self.taken_actions)


class Result:
    """Result object represents the model checking results for a given model

    Args:
        model: stormvogel representation of the model associated with the results
        values: for each state the model checking result
        scheduler: in case the model is an mdp we can optionally store a scheduler
    """

    model: stormvogel.model.Model
    # values are hashed by the state id:
    values: dict[int, stormvogel.model.Number]
    scheduler: Scheduler | None

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
        if self.scheduler is None:
            self.stormpy_scheduler = stormpy_scheduler
            taken_actions = {}
            for state in self.model.states.values():
                taken_actions[state.id] = self.model.get_action(
                    str(stormpy_scheduler.get_choice(state.id))
                )
            self.scheduler = Scheduler(self.model, taken_actions)
        else:
            print("This result already has a scheduler")

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Number | None:
        """returns the model checking result for a given state if present in the model"""
        if state in self.model.states.values():
            return self.values[state.id]
        else:
            print("This state is not a part of the model")
            return None

    def generate_induced_dtmc(self) -> stormvogel.model.Model | None:
        """Given an mdp that has a scheduler, this function creates and returns the scheduler induced markov chain"""
        if (
            self.model.get_type() == stormvogel.model.ModelType.MDP
            and self.scheduler is not None
        ):
            stormpy_mdp = stormvogel.map.stormvogel_to_stormpy(self.model)
            if stormpy_mdp is not None:
                stormpy_dtmc = stormpy_mdp.apply_scheduler(self.stormpy_scheduler)
                stormvogel_dtmc = stormvogel.map.stormpy_to_stormvogel(stormpy_dtmc)
                return stormvogel_dtmc
            else:
                print("something went wrong")
                return
        else:
            if self.scheduler is not None:
                print("This model is not an mdp")
            else:
                print("This result does not have a scheduler")
            return None

    def __str__(self) -> str:
        s = {}
        if self.scheduler is not None:
            for index, action in enumerate(self.scheduler.taken_actions.values()):
                s[index] = str(list(action.labels))

        add = ""
        if self.model.name is not None:
            add = "model: " + str(self.model.name) + "\n"
        return add + "values: \n " + str(self.values) + "\n" + "scheduler: \n " + str(s)


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
