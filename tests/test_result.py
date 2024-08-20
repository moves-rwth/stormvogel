import stormvogel.result
import stormpy


def test_convert_model_checker_results_dtmc():
    path = stormpy.examples.files.prism_dtmc_die
    prism_program = stormpy.parse_prism_program(path)
    formula_str = "P=? [F s=7 & d=2]"
    properties = stormpy.parse_properties(formula_str, prism_program)

    model = stormpy.build_model(prism_program, properties)
    result = stormpy.model_checking(model, properties[0])

    stormvogel_model = stormvogel.map.stormpy_to_stormvogel(model)
    if stormvogel_model is not None:
        stormvogel_result = stormvogel.result.convert_model_checking_result(
            stormvogel_model, result
        )
        assert list(stormvogel_result.values.values()) == result.get_values()
    else:
        assert False


"""
def test_convert_model_checker_results_mdp():
    path = stormpy.examples.files.prism_mdp_firewire
    prism_program = stormpy.parse_prism_program(path)
    #formula_str = "P=? [F s=7 & d=2]"
    #properties = stormpy.parse_properties(formula_str, prism_program)

    #model = stormpy.build_model(prism_program, properties)
    model = stormpy.build_model(prism_program)
    #result = stormpy.model_checking(model, properties[0])

    print(model)

    stormvogel_model = stormvogel.map.stormpy_to_stormvogel(model)
    #if stormvogel_model is not None:
    #    stormvogel.result.convert_model_checking_result(stormvogel_model, result)

    print(stormvogel_model)
"""
