import stormvogel.stormpy_utils.mapping as mapping
import stormvogel.parametric
import stormvogel.model
import stormvogel.examples.knuth_yao_pmc

try:
    import stormpy
except ImportError:
    stormpy = None


def test_pmc_conversion():
    if stormpy is not None:
        # Create a new model with the name "simple pmc"
        pmc = stormvogel.model.new_dtmc("simple pmc")

        init = pmc.get_initial_state()

        # From the initial state, we have two transitions that either bring us to state A or state B
        p1 = stormvogel.parametric.Polynomial(["x", "y", "z"])
        p1.add_term((1, 1, 1), 4)

        # the other transition is a rational function with two polynomials
        p2 = stormvogel.parametric.Polynomial(["x", "y"])
        # p3 = stormvogel.parametric.Polynomial(["z"])
        p2.add_term((2, 0), 1)
        p2.add_term((2, 2), -1)
        # p3.add_term((2,), 2)
        # r1 = stormvogel.parametric.RationalFunction(p2,p3)

        # TODO make it work for proper rational functions

        pmc.new_state(labels=["A"])
        pmc.new_state(labels=["B"])

        init.set_transitions(
            [
                (p1, pmc.get_states_with_label("A")[0]),
                (p2, pmc.get_states_with_label("B")[0]),
            ]
        )

        # we add self loops to all states with no outgoing transitions
        pmc.add_self_loops()

        # we test the mapping
        stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
        new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)

        assert pmc == new_pmc

        # We also test it for the knuth yao model
        pmc = stormvogel.examples.knuth_yao_pmc.create_knuth_yao_pmc()
        stormpy_pmc = mapping.stormvogel_to_stormpy(pmc)
        new_pmc = mapping.stormpy_to_stormvogel(stormpy_pmc)

        assert pmc == new_pmc


def test_pmdp_conversion():
    if stormpy is not None:
        # Create a new model with the name "simple pmdp"
        pmdp = stormvogel.model.new_mdp("simple pmdp")

        init = pmdp.get_initial_state()

        # From the initial state, we have two actions with transitions that either bring us to a goal state or sink state

        p1 = stormvogel.parametric.Polynomial(["x"])
        p2 = stormvogel.parametric.Polynomial(["x"])
        p1.add_term((1,), 1)
        p2.add_term((0,), 1)
        p2.add_term((1,), -1)

        goal = pmdp.new_state(labels=["goal"])
        sink = pmdp.new_state(labels=["sink"])

        action_a = pmdp.new_action(frozenset({"a"}))
        action_b = pmdp.new_action(frozenset({"b"}))
        branch0 = stormvogel.model.Branch(
            [
                (p1, goal),
                (p2, sink),
            ]
        )
        branch1 = stormvogel.model.Branch(
            [
                (p1, sink),
                (p2, goal),
            ]
        )

        pmdp.add_transitions(
            init, stormvogel.model.Transition({action_a: branch0, action_b: branch1})
        )

        # we add self loops to all states with no outgoing transitions
        pmdp.add_self_loops()

        # we test the mapping
        stormpy_pmdp = mapping.stormvogel_to_stormpy(pmdp)

        new_pmdp = mapping.stormpy_to_stormvogel(stormpy_pmdp)

        assert pmdp == new_pmdp


def test_pmc_valuations():
    # we build a simple pmc
    pmc = stormvogel.model.new_dtmc()

    init = pmc.get_initial_state()

    # From the initial state, we have two transitions that either bring us to state A or state B
    p1 = stormvogel.parametric.Polynomial(["x", "z", "w"])
    p1.add_term((1, 1, 2), 4)

    # the other transition is a rational function with two polynomials
    p2 = stormvogel.parametric.Polynomial(["x", "y"])
    p3 = stormvogel.parametric.Polynomial(["z"])
    p2.add_term((2, 0), 1)
    p2.add_term((2, 2), -1)
    p3.add_term((2,), 2)
    r1 = stormvogel.parametric.RationalFunction(p2, p3)

    pmc.new_state(labels=["A"])
    pmc.new_state(labels=["B"])

    init.set_transitions(
        [
            (p1, pmc.get_states_with_label("A")[0]),
            (r1, pmc.get_states_with_label("B")[0]),
        ]
    )

    # we add self loops to all states with no outgoing transitions
    pmc.add_self_loops()

    induced_pmc = pmc.parameter_valuation({"x": 1, "y": 2, "w": 1, "z": 5})

    # we build what the induced pmc is supposed to look like
    new_induced_pmc = stormvogel.model.new_dtmc()

    init = new_induced_pmc.get_initial_state()

    new_induced_pmc.new_state(labels=["A"])
    new_induced_pmc.new_state(labels=["B"])

    init.set_transitions(
        [
            (20, new_induced_pmc.get_states_with_label("A")[0]),
            (-0.06, new_induced_pmc.get_states_with_label("B")[0]),
        ]
    )

    # we add self loops to all states with no outgoing transitions
    new_induced_pmc.add_self_loops()

    assert induced_pmc == new_induced_pmc


def test_gradient_descent():
    if stormpy is not None:
        assert True


"""
def test_solution_function():
    if stormpy is not None:
        pmc = stormvogel.examples.knuth_yao_pmc.create_knuth_yao_pmc()

        # result = model_checking.model_checking(pmc, 'P=? [F "rolled1"]')
        # assert result is not None

        # print(result.values[0])



def test_stormpy_pmc():
    import stormpy
    import stormpy.info

    import stormpy.examples
    import stormpy.examples.files

    import stormpy._config as config

    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return

    import stormpy.pars

    if stormpy.info.storm_ratfunc_use_cln():
        from stormpy.pycarl.cln import formula
    else:
        from stormpy.pycarl.gmp import formula

    path = stormpy.examples.files.prism_pdtmc_die
    prism_program = stormpy.parse_prism_program(path)

    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    model = stormpy.build_parametric_model(prism_program, properties)

    print(model.transition_matrix)

    stormvogel_model = mapping.stormpy_to_stormvogel(model)
    model = mapping.stormvogel_to_stormpy(stormvogel_model)

    print(model.transition_matrix)

    initial_state = model.initial_states[0]
    result = stormpy.model_checking(model, properties[0])
    print("Result: {}".format(result.at(initial_state)))

    collector = stormpy.ConstraintCollector(model)
    print("Well formed constraints:")
    for formula in collector.wellformed_constraints:
        print(formula.get_constraint())
    print("Graph preserving constraints:")
    for formula in collector.graph_preserving_constraints:
        print(formula.get_constraint())
"""
