import stormpy.storage
import stormvogel.model


def build_matrix(model: stormvogel.model.Model) -> stormpy.storage.SparseMatrix:
    """
    Takes a model and creates a sparsematrix that represents the same transitions
    """
    builder = stormpy.storage.SparseMatrixBuilder()
    for transition in model.transitions.items():
        for branch in transition[1].transition.values():
            for tuple in branch.branch:
                builder.add_next_value(
                    row=transition[0], column=tuple[1].id, value=tuple[0]
                )
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


def stormvogel_to_stormpy(
    model: stormvogel.model.Model,
) -> stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp | None:
    def map_dtmc(model: stormvogel.model.Model) -> stormpy.storage.SparseDtmc:
        """
        Takes a simple representation of a dtmc as input and outputs a dtmc how it is represented in stormpy
        """

        # we first build the SparseMatrix
        matrix = build_matrix(model)

        # then we add the state labels
        state_labeling = add_labels(model)

        # TODO rewards

        # then we build the dtmc
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix, state_labeling=state_labeling
        )
        dtmc = stormpy.storage.SparseDtmc(components)

        return dtmc

    def map_mdp(model: stormvogel.model.Model) -> stormpy.storage.SparseMdp:
        """
        Takes a simple representation of an mdp as input and outputs an mdp how it is represented in stormpy
        """

        # we determine the number of choices and the choice labels
        count = 0
        for transition in model.transitions.items():
            for action in transition[1].transition.items():
                count += 1
        choice_labeling = stormpy.storage.ChoiceLabeling(count)

        if model.actions is not None:
            labels = model.actions.keys()
        else:
            labels = []

        for label in labels:
            choice_labeling.add_label(label)

        # then we create the matrix and simultanuously add the correct labels to the choices
        builder = stormpy.SparseMatrixBuilder(
            rows=0,
            columns=0,
            entries=0,
            force_dimensions=False,
            has_custom_row_grouping=True,
            row_groups=0,
        )
        row_index = 0
        for transition in model.transitions.items():
            builder.new_row_group(row_index)
            for action in transition[1].transition.items():
                for tuple in action[1].branch:
                    builder.add_next_value(
                        row_index, column=tuple[1].id, value=tuple[0]
                    )

                if not action[0] == stormvogel.model.EmptyAction:
                    choice_labeling.add_label_to_choice(action[0].name, row_index)
                row_index += 1

        matrix = builder.build()

        # then we add the state labels
        state_labeling = add_labels(model)

        # TODO rewards

        # then we build the mdp
        components = stormpy.SparseModelComponents(
            transition_matrix=matrix, state_labeling=state_labeling
        )
        components.choice_labeling = choice_labeling
        mdp = stormpy.storage.SparseMdp(components)

        return mdp

    if model.get_type() == stormvogel.model.ModelType(1):
        return map_dtmc(model)
    elif model.get_type() == stormvogel.model.ModelType(2):
        return map_mdp(model)
    else:
        print("This type of model is not yet supported for this action")
        return None


def stormpy_to_stormvogel(
    sparsemodel: stormpy.storage.SparseDtmc | stormpy.storage.SparseMdp,
) -> stormvogel.model.Model:
    def map_dtmc(sparsedtmc: stormpy.storage.SparseDtmc) -> stormvogel.model.Model:
        """
        Takes a dtmc stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = stormvogel.model.new_dtmc(name=None)

        # we add the states
        for state in sparsedtmc.states:
            # the initial state is automatically added so we don't add it
            if state.id > 0:
                model.new_state(labels=list(state.labels))

        # we add the transitions
        matrix = sparsedtmc.transition_matrix
        for state in sparsedtmc.states:
            # we add the transitions
            row = matrix.get_row(state.id)
            transitionshorthand = [
                (x.value(), model.get_state_by_id(x.column)) for x in row
            ]
            transitions = stormvogel.model.transition_from_shorthand(
                transitionshorthand
            )
            model.set_transitions(model.get_state_by_id(state.id), transitions)

        # TODO rewards

        return model

    def map_mdp(sparsemdp: stormpy.storage.SparseDtmc) -> stormvogel.model.Model:
        """
        Takes a mdp stormpy representation as input and outputs a simple stormvogel representation
        """

        # we create the model
        model = stormvogel.model.new_mdp(name=None)

        # we add the states
        for state in sparsemdp.states:
            # the initial state is automatically added so we don't add it
            if state.id > 0:
                model.new_state(labels=list(state.labels))

        # we add the transitions
        matrix = sparsemdp.transition_matrix
        for index, state in enumerate(sparsemdp.states):
            row_group_start = matrix.get_row_group_start(index)
            row_group_end = matrix.get_row_group_end(index)
            # within a row group we add for each action the transitions
            transition = dict()

            for i in range(row_group_start, row_group_end):
                row = matrix.get_row(i)

                # actionlabels = sparsemdp.choice_labeling.get_labels_of_choice(i)
                # actionlabelslist = [str(x) for x in actionlabels]
                # print(actionlabelslist)

                # for now assign a name based on index
                # TODO assign the correct labels and name
                action = stormvogel.model.Action(str(i))

                branch = [(x.value(), model.get_state_by_id(x.column)) for x in row]

                transition[action] = stormvogel.model.Branch(branch)

                transitions = stormvogel.model.Transition(transition)

                model.set_transitions(model.get_state_by_id(state.id), transitions)

        # TODO rewards

        return model

    if sparsemodel.transition_matrix.has_trivial_row_grouping:
        return map_dtmc(sparsemodel)
    else:
        return map_mdp(sparsemodel)


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
