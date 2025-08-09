from fractions import Fraction
from stormvogel.model import value_to_string, Interval
from stormvogel import parametric


def test_int():
    assert value_to_string(46, False, 4, 1000) == "46.0"
    assert value_to_string(46, True, 4, 1000) == "46"


def test_float():
    assert value_to_string(46.0, False, 4, 1000) == "46.0"
    assert value_to_string(46.0, True, 4, 1000) == "46"
    assert value_to_string(0.66666666, False, 4, 1000) == "0.6667"


def test_fraction():
    assert value_to_string(Fraction(30, 40), True, 4, 1000) == "3/4"
    assert value_to_string(Fraction(30, 40), False, 4, 1000) == "0.75"
    assert value_to_string(Fraction(1, 3), False, 4, 1000) == "0.3333"
    assert value_to_string(Fraction(1, 1500), True, 4, 1000) == "1/1000"


def test_parametric_polynomial():
    pol = parametric.Polynomial(["x", "y", "z"])
    pol.add_term((1, 2, 3), 4)
    pol.add_term((1, 0, 0), 3)
    assert value_to_string(pol) == "4.0*xy^2z^3 + 3.0*x"


def test_parametric_rational():
    pol1 = parametric.Polynomial(["x", "y", "z"])
    pol1.add_term((1, 2, 3), 4)
    pol1.add_term((1, 0, 0), 3)
    pol2 = parametric.Polynomial(["z"])
    pol2.add_term((1,), 2)
    rat = parametric.RationalFunction(pol1, pol2)
    assert value_to_string(rat) == "(4.0*xy^2z^3 + 3.0*x)/(2.0*z)"


def test_interval():
    itvl = Interval(Fraction(1, 3), Fraction(5, 8))
    assert value_to_string(itvl) == "[1/3,5/8]"
