import stormpy.storage
import stormvogel.model


def stormvogel_to_stormpy(
    model: stormvogel.model.Model,
) -> (
    stormpy.storage.SparseDtmc
    | stormpy.storage.SparseMdp
    | stormpy.storage.SparseCtmc
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
                    # print(str(list(action[0].labels)))
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
        for transition in model.transitions.items():
            for action in transition[1].transition.items():
                count += 1
                if not action[0] == stormvogel.model.EmptyAction:
                    for label in action[0].labels:
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

        # we first build the SparseMatrix
        matrix = build_matrix(model, None)

        # then we add the state labels
        state_labeling = add_labels(model)

        # then we add the rewards
        reward_models = add_rewards(model)

        # then we build the dtmc and we add the exit rates if necessary
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix,
            state_labeling=state_labeling,
            reward_models=reward_models,
            rate_transitions=True,  # for now we always set this to True since we always work with rate transitions in stormvogel
        )
        if not model.rates == {} and model.rates is not None:
            components.exit_rates = list(model.rates.values())

        ctmc = stormpy.storage.SparseCtmc(components)

        return ctmc

    # we check the type to handle the model correctly
    if model.get_type() == stormvogel.model.ModelType.DTMC:
        return map_dtmc(model)
    elif model.get_type() == stormvogel.model.ModelType.MDP:
        return map_mdp(model)
    elif model.get_type() == stormvogel.model.ModelType.CTMC:
        return map_ctmc(model)
    else:
        print("This type of model is not yet supported for this action")
        return None


def stormpy_to_stormvogel(
    sparsemodel: stormpy.storage.SparseDtmc
    | stormpy.storage.SparseMdp
    | stormpy.storage.SparseCtmc,
) -> stormvogel.model.Model | None:
    def add_states(
        model: stormvogel.model.Model,
        sparsemodel: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
    ):
        """
        helper function to add the states from the sparsemodel to the model
        """
        for state in sparsemodel.states:
            # the initial state is automatically added so we don't add it
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

                # TODO assign the correct action name and not only an index
                actionlabels = frozenset(
                    sparsemdp.choice_labeling.get_labels_of_choice(i)
                    if sparsemdp.has_choice_labeling()
                    else str(i)
                )
                action = model.new_action_with_labels(str(i), actionlabels)
                branch = [(x.value(), model.get_state_by_id(x.column)) for x in row]
                transition[action] = stormvogel.model.Branch(branch)
                transitions = stormvogel.model.Transition(transition)
                model.set_transitions(model.get_state_by_id(state.id), transitions)

        # we add the reward models to the state action pairs
        add_rewards(model, sparsemdp)

        return model

    def map_ctmc(sparsectmc: stormpy.storage.SparseCtmc) -> stormvogel.model.Model:
        """
        Takes a ctmc stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model (it seems names are not stored in sparsedtmcs)
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

        # we add the reward models to the states
        add_rewards(model, sparsectmc)

        # we set the correct exit rates
        for state in model.states.items():
            model.set_rate(state[1], sparsectmc.exit_rates[0])

        return model

    # we check the type to handle the sparse model correctly
    if sparsemodel.model_type.name == "DTMC":
        return map_dtmc(sparsemodel)
    elif sparsemodel.model_type.name == "MDP":
        return map_mdp(sparsemodel)
    elif sparsemodel.model_type.name == "CTMC":
        return map_ctmc(sparsemodel)
    else:
        print("This type of model is not yet supported for this action")
        return


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
