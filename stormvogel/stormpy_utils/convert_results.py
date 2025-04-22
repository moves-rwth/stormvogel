import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.model
import stormvogel.result

try:
    import stormpy
except ImportError:
    stormpy = None



def generate_induced_dtmc(model: stormvogel.model.Model, scheduler: stormvogel.result.Scheduler) -> stormvogel.model.Model | None:
    """Given an mdp that has a scheduler, this function creates and returns the scheduler induced markov chain"""
    if (
        model.get_type() == stormvogel.model.ModelType.MDP
        and scheduler is not None
    ):
        stormpy_mdp = mapping.stormvogel_to_stormpy(model)
        stormpy_scheduler = convert_scheduler_to_stormpy(model, scheduler)
        stormpy_dtmc = stormpy_mdp.apply_scheduler(stormpy_scheduler)
        stormvogel_dtmc = mapping.stormpy_to_stormvogel(stormpy_dtmc)
        return stormvogel_dtmc
    else:
        if self.scheduler is not None:
            raise RuntimeError("This model is not an mdp")
        else:
            raise RuntimeError("This result does not have a scheduler")


def convert_scheduler_to_stormpy(model: stormvogel.model.Model, scheduler: stormvogel.result.Scheduler):
    #print(dir(stormpy.storage.Scheduler))

    stormpy_model = mapping.stormvogel_to_stormpy(model)
    stormpy_scheduler = stormpy.storage.Scheduler(stormpy_model) #this does not work yet because there is no constructor

    return stormpy_scheduler


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

    return stormvogel.result.Scheduler(model, taken_actions)


def convert_model_checking_result(
    model: stormvogel.model.Model,
    stormpy_result: stormpy.core.ExplicitQuantitativeCheckResult
    | stormpy.core.ExplicitQualitativeCheckResult
    | stormpy.core.ExplicitParametricQuantitativeCheckResult,
    with_scheduler: bool = True,
) -> stormvogel.result.Result | None:
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
            stormvogel_result = stormvogel.result.Result(
                model,
                {
                    index: value
                    for (index, value) in enumerate(stormpy_result.get_values())
                },
                scheduler=convert_scheduler_to_stormvogel(
                    model, stormpy_result.scheduler
                ),
            )
        else:
            stormvogel_result = stormvogel.result.Result(
                model,
                {
                    index: value
                    for (index, value) in enumerate(stormpy_result.get_values())
                },
            )
    elif type(stormpy_result == stormpy.core.ExplicitQualitativeCheckResult):
        values = {i: stormpy_result.at(i) for i in range(0, len(model.states))}
        if stormpy_result.has_scheduler and with_scheduler:
            stormvogel_result = stormvogel.result.Result(
                model,
                values,
                scheduler=convert_scheduler_to_stormvogel(
                    model, stormpy_result.scheduler
                ),
            )
        else:
            stormvogel_result = stormvogel.result.Result(model, values)
    else:
        raise RuntimeError("Unsupported result type")

    return stormvogel_result
