import stormpy.storage
import stormvogel.model


def stormvogel_to_stormpy(
    model: stormvogel.model.Model,
) -> (
    stormpy.storage.SparseDtmc
    | stormpy.storage.SparseMdp
    | stormpy.storage.SparseCtmc
    | stormpy.storage.SparsePomdp
    | None
):
    def build_matrix(
        model: stormvogel.model.Model,
        choice_labeling: stormpy.storage.ChoiceLabeling | None,
    ) -> stormpy.storage.SparseMatrix:
        """
        Takes a model and creates a stormpy sparsematrix that represents the same transitions
        """
        row_grouping = model.supports_actions()
        builder = stormpy.SparseMatrixBuilder(
            rows=0,
            columns=0,
            entries=0,
            force_dimensions=False,
            has_custom_row_grouping=row_grouping,
            row_groups=0,
        )
        row_index = 0
        for transition in model.transitions.items():
            if row_grouping:
                builder.new_row_group(row_index)
            for action in transition[1].transition.items():
                for tuple in action[1].branch:
                    builder.add_next_value(
                        row=row_index, column=tuple[1].id, value=tuple[0]
                    )

                # if there is an action then add the label to the choice
                if (
                    not action[0] == stormvogel.model.EmptyAction
                    and choice_labeling is not None
                ):
                    for label in action[0].labels:
                        choice_labeling.add_label_to_choice(str(label), row_index)
                row_index += 1

        matrix = builder.build()
        return matrix

    def add_labels(model: stormvogel.model.Model) -> stormpy.storage.StateLabeling:
        """
        Takes a model creates a state labelling object that determines which states get which labels in the stormpy representation
        """

        state_labeling = stormpy.storage.StateLabeling(len(model.states))

        # we first add all the different labels
        for label in model.get_labels():
            state_labeling.add_label(label)

        # then we assign the labels to the correct states
        for state in model.states.items():
            for label in state[1].labels:
                state_labeling.add_label_to_state(label, state[0])

        return state_labeling

    def add_rewards(
        model: stormvogel.model.Model,
    ) -> dict[str, stormpy.SparseRewardModel]:
        """
        Takes a model and creates a dictionary of all the stormpy representations of reward models
        """
        reward_models = {}
        for rewardmodel in model.rewards:
            reward_models[rewardmodel.name] = stormpy.SparseRewardModel(
                optional_state_action_reward_vector=list(rewardmodel.rewards.values())
            )

        return reward_models

    def map_dtmc(model: stormvogel.model.Model) -> stormpy.storage.SparseDtmc:
        """
        Takes a simple representation of a dtmc as input and outputs a dtmc how it is represented in stormpy
        """

        # we first build the SparseMatrix
        matrix = build_matrix(model, None)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # then we build the dtmc
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        dtmc = stormpy.storage.SparseDtmc(components)

        return dtmc

    def map_mdp(model: stormvogel.model.Model) -> stormpy.storage.SparseMdp:
        """
        Takes a simple representation of an mdp as input and outputs an mdp how it is represented in stormpy
        """

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model.states.values():
            for action in state.available_actions():
                count += 1
                if not action == stormvogel.model.EmptyAction:
                    for label in action.labels:
                        labels.add(label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # then we build the mdp
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseMdp(components)

        return mdp

    def map_ctmc(model: stormvogel.model.Model) -> stormpy.storage.SparseCtmc:
        """
        Takes a simple representation of a ctmc as input and outputs a ctmc how it is represented in stormpy
        """

        # we first build the SparseMatrix (in stormvogel these are always the rate transitions)
        matrix = build_matrix(model, None)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # then we build the ctmc and we add the exit rates if necessary
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            rate_transitions=True,
        )
        if not model.exit_rates == {} and model.exit_rates is not None:
            components.exit_rates = list(model.exit_rates.values())

        ctmc = stormpy.storage.SparseCtmc(components)

        return ctmc

    def map_pomdp(model: stormvogel.model.Model) -> stormpy.storage.SparsePomdp:
        """
        Takes a simple representation of an pomdp as input and outputs an pomdp how it is represented in stormpy
        """

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model.states.values():
            for action in state.available_actions():
                count += 1
                if not action == stormvogel.model.EmptyAction:
                    for label in action.labels:
                        labels.add(label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # then we build the pomdp
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
        )
        components.observability_classes = list(
            [
                state.get_observation().get_observation()
                for state in model.states.values()
            ]
        )
        components.choice_labeling = choice_labeling
        pomdp = stormpy.storage.SparsePomdp(components)

        return pomdp

    def map_ma(model: stormvogel.model.Model) -> stormpy.storage.SparseMA:
        """
        Takes a simple representation of an ma as input and outputs an ma how it is represented in stormpy
        """

        # we determine the number of choices and the labels
        count = 0
        labels = set()
        for state in model.states.values():
            for action in state.available_actions():
                count += 1
                if not action == stormvogel.model.EmptyAction:
                    for label in action.labels:
                        labels.add(label)

        # we add the labels to the choice labeling object
        choice_labeling = stormpy.storage.ChoiceLabeling(count)
        for label in labels:
            choice_labeling.add_label(str(label))

        # then we create the matrix and simultanuously add the correct labels to the choices
        matrix = build_matrix(model, choice_labeling=choice_labeling)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # we create the list of markovian state ids
        assert model.markovian_states is not None
        markovian_states_list = [state.id for state in model.markovian_states]
        if isinstance(markovian_states_list, list):
            markovian_states_bitvector = stormpy.storage.BitVector(
                max(markovian_states_list) + 1,
                markovian_states_list,
            )
        else:
            markovian_states_bitvector = stormpy.storage.BitVector(0)

        # then we build the ma
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            markovian_states=markovian_states_bitvector,
        )
        if not model.exit_rates == {} and model.exit_rates is not None:
            components.exit_rates = list(model.exit_rates.values())
        else:
            components.exit_rates = []
        components.choice_labeling = choice_labeling
        ma = stormpy.storage.SparseMA(components)

        return ma

    if model.all_states_outgoing_transition():
        # we check the type to handle the model correctly
        if model.get_type() == stormvogel.model.ModelType.DTMC:
            return map_dtmc(model)
        elif model.get_type() == stormvogel.model.ModelType.MDP:
            return map_mdp(model)
        elif model.get_type() == stormvogel.model.ModelType.CTMC:
            return map_ctmc(model)
        elif model.get_type() == stormvogel.model.ModelType.POMDP:
            return map_pomdp(model)
        elif model.get_type() == stormvogel.model.ModelType.MA:
            return map_ma(model)
        else:
            raise RuntimeError(
                "This type of model is not yet supported for this action"
            )
    else:
        raise RuntimeError(
            "This model has states with no outgoing transitions.\nUse the add_self_loops() function to add self loops to all states with no outgoing transition."
        )


def stormpy_to_stormvogel(
    sparsemodel: stormpy.storage.SparseDtmc
    | stormpy.storage.SparseMdp
    | stormpy.storage.SparseCtmc
    | stormpy.storage.SparsePomdp
    | stormpy.storage.SparseMA,
) -> stormvogel.model.Model | None:
    def add_states(
        model: stormvogel.model.Model,
        sparsemodel: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    ):
        """
        helper function to add the states from the sparsemodel to the model
        """
        for state in sparsemodel.states:
            if state.id == 0:
                for label in state.labels:
                    model.get_state_by_id(0).add_label(label)
            if state.id > 0:
                model.new_state(labels=list(state.labels))

    def add_rewards(
        model: stormvogel.model.Model,
        sparsemodel: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    ):
        """
        adds the rewards from the sparsemodel to either the states or the state action pairs of the model
        """
        for reward_model in sparsemodel.reward_models:
            rewards = sparsemodel.get_reward_model(reward_model)
            rewardmodel = model.add_rewards(reward_model)
            for index, reward in enumerate(
                rewards.state_action_rewards
                if rewards.has_state_action_rewards
                else rewards.state_rewards
            ):
                if model.supports_actions():
                    rewardmodel.set_action_state(index, reward)
                else:
                    rewardmodel.set(model.get_state_by_id(index), reward)

    def map_dtmc(sparsedtmc: stormpy.storage.SparseDtmc) -> stormvogel.model.Model:
        """
        Takes a dtmc stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model (it seems names are not stored in sparsedtmcs)
        model = stormvogel.model.new_dtmc(name=None)

        # we add the states
        add_states(model, sparsedtmc)

        # we add the transitions
        matrix = sparsedtmc.transition_matrix
        for state in sparsedtmc.states:
            row = matrix.get_row(state.id)
            transitionshorthand = [
                (x.value(), model.get_state_by_id(x.column)) for x in row
            ]
            transitions = stormvogel.model.transition_from_shorthand(
                transitionshorthand
            )
            model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add self loops to all states with no outgoing transition
        model.add_self_loops()

        # we add the reward models to the states
        add_rewards(model, sparsedtmc)

        return model

    def map_mdp(sparsemdp: stormpy.storage.SparseDtmc) -> stormvogel.model.Model:
        """
        Takes a mdp stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = stormvogel.model.new_mdp(name=None)

        # we add the states
        add_states(model, sparsemdp)

        # we add the transitions
        matrix = sparsemdp.transition_matrix
        for index, state in enumerate(sparsemdp.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            transition = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                actionlabels = frozenset(
                    sparsemdp.choice_labeling.get_labels_of_choice(i)
                    if sparsemdp.has_choice_labeling()
                    else str(i)
                )
                # TODO assign the correct action name and not only an index
                action = model.new_action(str(i), actionlabels)
                branch = [(x.value(), model.get_state_by_id(x.column)) for x in row]
                transition[action] = stormvogel.model.Branch(branch)
                transitions = stormvogel.model.Transition(transition)
                model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        add_rewards(model, sparsemdp)

        return model

    def map_ctmc(sparsectmc: stormpy.storage.SparseCtmc) -> stormvogel.model.Model:
        """
        Takes a ctmc stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model (it seems names are not stored in sparsectmcs)
        model = stormvogel.model.new_ctmc(name=None)

        # we add the states
        add_states(model, sparsectmc)

        # we add the transitions
        matrix = sparsectmc.transition_matrix
        for state in sparsectmc.states:
            row = matrix.get_row(state.id)
            transitionshorthand = [
                (x.value(), model.get_state_by_id(x.column)) for x in row
            ]
            transitions = stormvogel.model.transition_from_shorthand(
                transitionshorthand
            )
            model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the states
        add_rewards(model, sparsectmc)

        # we set the correct exit rates
        for state in model.states.items():
            model.set_rate(state[1], sparsectmc.exit_rates[state[1].id])

        return model

    def map_pomdp(sparsepomdp: stormpy.storage.SparsePomdp) -> stormvogel.model.Model:
        """
        Takes a pomdp stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model (it seems names are not stored in sparsepomdps)
        model = stormvogel.model.new_pomdp(name=None)

        # we add the states
        add_states(model, sparsepomdp)

        # we add the transitions
        matrix = sparsepomdp.transition_matrix
        for index, state in enumerate(sparsepomdp.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            transition = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                actionlabels = frozenset(
                    sparsepomdp.choice_labeling.get_labels_of_choice(i)
                    if sparsepomdp.has_choice_labeling()
                    else str(i)
                )
                # TODO assign the correct action name and not only an index
                action = model.new_action(str(i), actionlabels)
                branch = [(x.value(), model.get_state_by_id(x.column)) for x in row]
                transition[action] = stormvogel.model.Branch(branch)
                transitions = stormvogel.model.Transition(transition)
                model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        add_rewards(model, sparsepomdp)

        # we add the observations:
        for state in model.states.values():
            state.new_observation(sparsepomdp.get_observation(state.id))

        return model

    def map_ma(sparsema: stormpy.storage.SparseMA) -> stormvogel.model.Model:
        """
        Takes a ma stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model (it seems names are not stored in sparsemas)
        model = stormvogel.model.new_ma(name=None)

        # we add the states
        add_states(model, sparsema)

        # we add the transitions
        matrix = sparsema.transition_matrix
        for index, state in enumerate(sparsema.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)

            # within a row group we add for each action the transitions
            transition = dict()
            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                actionlabels = frozenset(
                    sparsema.choice_labeling.get_labels_of_choice(i)
                    if sparsema.has_choice_labeling()
                    else str(i)
                )
                # TODO assign the correct action name and not only an index
                action = model.new_action(str(i), actionlabels)
                branch = [(x.value(), model.get_state_by_id(x.column)) for x in row]
                transition[action] = stormvogel.model.Branch(branch)
                transitions = stormvogel.model.Transition(transition)
                model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add self loops to all states with no outgoing transitions
        model.add_self_loops()

        # we add the reward models to the state action pairs
        add_rewards(model, sparsema)

        # we set the correct exit rates
        for state in model.states.items():
            model.set_rate(state[1], sparsema.exit_rates[state[1].id])

        # we set the markovian states
        for state_id in list(sparsema.markovian_states):
            model.add_markovian_state(model.get_state_by_id(state_id))

        return model

    # we check the type to handle the sparse model correctly
    if sparsemodel.model_type.name == "DTMC":
        return map_dtmc(sparsemodel)
    elif sparsemodel.model_type.name == "MDP":
        return map_mdp(sparsemodel)
    elif sparsemodel.model_type.name == "CTMC":
        return map_ctmc(sparsemodel)
    elif sparsemodel.model_type.name == "POMDP":
        return map_pomdp(sparsemodel)
    elif sparsemodel.model_type.name == "MA":
        return map_ma(sparsemodel)
    else:
        raise RuntimeError("This type of model is not yet supported for this action")


if __name__ == "__main__":
    dtmc = stormvogel.model.new_dtmc("Die")
    init = dtmc.get_initial_state()
    init.set_transitions(
        [(1 / 6, dtmc.new_state(f"rolled{i}", {"rolled": i})) for i in range(6)]
    )

    print(dtmc)
    sparsedtmc = stormvogel_to_stormpy(dtmc)
    print(sparsedtmc)

    new_dtmc = stormpy_to_stormvogel(sparsedtmc)
    print(new_dtmc)
