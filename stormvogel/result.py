import stormvogel.stormpy_utils.mapping as mapping
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
        self, state: stormvogel.model.State | int
    ) -> stormvogel.model.Action:
        """returns the choice in the scheduler for the given state if present in the model"""
        if isinstance(state, int):
            state = self.model.get_state_by_id(state)
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
        values: dict[int, stormvogel.model.Number],
        scheduler: Scheduler | stormpy.storage.Scheduler | None = None,
    ):
        self.model = model
        self.values = values

        assert stormpy is not None
        if isinstance(scheduler, stormpy.storage.Scheduler):
            self.scheduler = convert_scheduler_to_stormvogel(self.model, scheduler)
            self.stormpy_scheduler = scheduler
        elif isinstance(scheduler, Scheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

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
            stormpy_mdp = mapping.stormvogel_to_stormpy(self.model)
            if stormpy_mdp is not None:
                stormpy_dtmc = stormpy_mdp.apply_scheduler(self.stormpy_scheduler)
                stormvogel_dtmc = mapping.stormpy_to_stormvogel(stormpy_dtmc)
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


def convert_scheduler_to_stormvogel(
    model: stormvogel.model.Model, stormpy_scheduler: stormpy.storage.Scheduler
):
    """Converts a stormpy scheduler to a stormvogel scheduler"""
    taken_actions = {}
    for state in model.states.values():
        av_act = state.available_actions()
        choice = stormpy_scheduler.get_choice(state.id)
        action_index = choice.get_deterministic_choice()
        taken_actions[state.id] = av_act[action_index]

    return Scheduler(model, taken_actions)


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
        if stormpy_result.has_scheduler and with_scheduler:
            stormvogel_result = Result(
                model,
                {
                    index: value
                    for (index, value) in enumerate(stormpy_result.get_values())
                },
                scheduler=stormpy_result.scheduler,
            )
        else:
            stormvogel_result = Result(
                model,
                {
                    index: value
                    for (index, value) in enumerate(stormpy_result.get_values())
                },
            )
    elif type(stormpy_result == stormpy.core.ExplicitQualitativeCheckResult):
        values = {i: stormpy_result.at(i) for i in range(0, len(model.states))}
        if stormpy_result.has_scheduler and with_scheduler:
            stormvogel_result = Result(
                model, values, scheduler=stormpy_result.scheduler
            )
        else:
            stormvogel_result = Result(model, values)
    else:
        raise RuntimeError("Unsupported result type")

    return stormvogel_result
