from fractions import Fraction
from stormvogel.model import number_to_string


def test_int():
    assert number_to_string(46, False, 4, 1000) == "46.0"
    assert number_to_string(46, True, 4, 1000) == "46"


def test_float():
    assert number_to_string(46.0, False, 4, 1000) == "46.0"
    assert number_to_string(46.0, True, 4, 1000) == "46"
    assert number_to_string(0.66666666, False, 4, 1000) == "0.6667"


def test_fraction():
    assert number_to_string(Fraction(30, 40), True, 4, 1000) == "3/4"
    assert number_to_string(Fraction(30, 40), False, 4, 1000) == "0.75"
    assert number_to_string(Fraction(1, 3), False, 4, 1000) == "0.3333"
    assert number_to_string(Fraction(1, 1500), True, 4, 1000) == "1/1000"


def test_parametric_polynomial():
    # TODO Pim can you help me?
    pass


def test_parametric_rational():
    # TODO Pim can you help me?
    pass
