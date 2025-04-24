import stormvogel.model


class Scheduler:
    """
    Scheduler object specifies what action to take in each state

    Args:
        model: mdp model associated with the scheduler
        taken_actions: for each state the action we choose in that state
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

        # TODO functionality to convert a lambda scheduler to this object

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

    def generate_induced_dtmc(self) -> stormvogel.model.Model | None:
        """This function resolves the nondeterminacy of the mdp and returns the scheduler induced dtmc"""
        if self.model.get_type() == stormvogel.model.ModelType.MDP:
            induced_dtmc = stormvogel.model.new_dtmc(create_initial_state=False)
            for state in self.model.states.values():
                induced_dtmc.new_state(labels=state.labels, valuations=state.valuations)
                action = self.get_choice_of_state(state)
                transitions = state.get_outgoing_transitions(action)
                assert transitions is not None
                induced_dtmc.add_transitions(s=state, transitions=transitions)

            return induced_dtmc

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
        scheduler: Scheduler | None = None,
    ):
        self.model = model
        self.values = values

        if isinstance(scheduler, Scheduler):
            self.scheduler = scheduler
        else:
            self.scheduler = None

    def get_result_of_state(
        self, state: stormvogel.model.State
    ) -> stormvogel.model.Number | None:
        """returns the model checking result for a given state"""
        if state in self.model.states.values():
            return self.values[state.id]
        else:
            raise RuntimeError("This state is not a part of the model")

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
