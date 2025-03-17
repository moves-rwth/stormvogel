import stormvogel.mapping
import stormvogel.model

try:
    import stormpy
except ImportError:
    stormpy = None


class Scheduler:
    """
    Scheduler object specifiec what action to take in each state

    Args:
        model: stormvogel representation of the model associated with the scheduler
        taken_actions: for each state an action to take in that state
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
            raise RuntimeError("This state is not a part of the model")

    def __str__(self) -> str:
        if self.model.name is not None:
            add = "Scheduler for model: " + self.model.name + "\n"
        else:
            add = ""
        return add + "taken actions: " + str(self.taken_actions)

    def __eq__(self, other) -> bool:
        if isinstance(other, Scheduler):
            return self.taken_actions == other.taken_actions
        return False


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
        self,
        model: stormvogel.model.Model,
        values: list[stormvogel.model.Number],
        scheduler: Scheduler | None = None,
    ):
        self.model = model
        self.scheduler = scheduler
        self.values = {}
        for index, val in enumerate(values):
            self.values[index] = val

    # TODO move function out of class
    def add_scheduler(self, stormpy_scheduler: stormpy.storage.Scheduler):
        """adds a scheduler to the result"""
        if self.scheduler is None:
            self.stormpy_scheduler = stormpy_scheduler
            taken_actions = {}
            for state in self.model.states.values():
                # taken_actions[state.id] = stormvogel.model.Action.create(
                #     str(stormpy_scheduler.get_choice(state.id))
                # )
                av_act = state.available_actions()
                choice = stormpy_scheduler.get_choice(state.id)
                action_index = choice.get_deterministic_choice()

                if len(av_act) == action_index:
                    print(len(self.model.states))
                    print(state)
                    print(av_act)
                    print(action_index)
                    print(dir(stormpy_scheduler))

                taken_actions[state.id] = av_act[action_index]

            self.scheduler = Scheduler(self.model, taken_actions)
        else:
            raise RuntimeError("This result already has a scheduler")

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Number | None:
        """returns the model checking result for a given state if present in the model"""
        if state in self.model.states.values():
            return self.values[state.id]
        else:
            raise RuntimeError("This state is not a part of the model")

    def generate_induced_dtmc(self) -> stormvogel.model.Model | None:
        """Given an mdp that has a scheduler, this function creates and returns the scheduler induced markov chain"""
        if (
            self.model.get_type() == stormvogel.model.ModelType.MDP
            and self.scheduler is not None
        ):
            stormpy_mdp = stormvogel.mapping.stormvogel_to_stormpy(self.model)
            if stormpy_mdp is not None:
                stormpy_dtmc = stormpy_mdp.apply_scheduler(self.stormpy_scheduler)
                stormvogel_dtmc = stormvogel.mapping.stormpy_to_stormvogel(stormpy_dtmc)
                return stormvogel_dtmc
            else:
                raise RuntimeError("Something went wrong")
        else:
            if self.scheduler is not None:
                raise RuntimeError("This model is not an mdp")
            else:
                raise RuntimeError("This result does not have a scheduler")

    def __str__(self) -> str:
        add = ""
        if self.model.name is not None:
            add = "model: " + str(self.model.name) + "\n"
        return (
            add
            + "values: \n "
            + str(self.values)
            + "\n"
            + "scheduler: \n "
            + str(self.scheduler)
        )

    def __eq__(self, other) -> bool:
        if isinstance(other, Result):
            return self.values == other.values and self.scheduler == other.scheduler
        return False


def convert_model_checking_result(
    model: stormvogel.model.Model,
    stormpy_result: stormpy.core.ExplicitQuantitativeCheckResult
    | stormpy.core.ExplicitQualitativeCheckResult
    | stormpy.core.ExplicitParametricQuantitativeCheckResult,
    with_scheduler: bool = True,
) -> Result | None:
    """
    Takes a model checking result from stormpy and its associated model and converts it to a stormvogel representation
    """
    assert stormpy is not None

    if (
        type(stormpy_result) == stormpy.core.ExplicitQuantitativeCheckResult
        or type(stormpy_result)
        == stormpy.core.ExplicitParametricQuantitativeCheckResult
    ):
        stormvogel_result = Result(model, stormpy_result.get_values())
    elif type(stormpy_result == stormpy.core.ExplicitQualitativeCheckResult):
        values = [stormpy_result.at(i) for i in range(0, len(model.states))]
        stormvogel_result = Result(model, values)
    else:
        raise RuntimeError("Unsupported result type")

    if stormpy_result.has_scheduler and with_scheduler:
        stormvogel_result.add_scheduler(stormpy_result.scheduler)

    return stormvogel_result
