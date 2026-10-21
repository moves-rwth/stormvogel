def example_parametric_models_01():
    import stormpy.examples
    import stormpy.examples.files

    path = stormpy.examples.files.prism_pdtmc_die
    prism_program = stormpy.parse_prism_program(path)

    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)
    model = stormpy.build_model[stormpy.RationalFunction](prism_program, properties)

    return model
