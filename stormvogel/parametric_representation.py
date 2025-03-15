import numpy as np


class Polynomial:
    """
    Represents polynomials, to be used as values for parametric models.
    Polynomials are represented as an n-dimensional (numpy array) tensor.

    Args:
        coefficients: coefficients of the terms
    """

    coefficients: np.ndarray = np.array([])

    def __init__(self, degree: int, dimension: int):
        self.coefficients = np.zeros((degree + 1,) * dimension)

    def set_coefficient(self, exponents: tuple[int, ...], coefficient: float):
        self.coefficients[exponents] = coefficient

    # TODO valuation function

    def __str__(self) -> str:
        s = ""
        # we iterate through each term
        for index, id in enumerate(np.ndindex(self.coefficients.shape)):
            # we only print terms with nonzero coefficients
            if self.coefficients[id] != 0:
                # we don't print coefficients that are 1
                if self.coefficients[id] != 1:
                    s += f"{self.coefficients[id]}*"

                # we print the variables with their corresponding powers
                # if the tuple only consists of zeroes then we are left with 1
                all_zero = True
                for i in range(len(id)):
                    if id[i] != 0:
                        all_zero = False
                        s += f"x_{i}"
                        if id[i] != 1:
                            s += f"^{id[i]}"
                if all_zero:
                    s += "1"
                s += " + "

        return s[:-3]


class RationalFunction:
    """
    Represents rational functions, to be used as values for parametric models
    Rational functions are represented as a pair of polynomials.

     Args:
        numerator: Polynomial in the numerator
        denominator: Polynomial in the denominator
    """

    numerator: Polynomial
    denominator: Polynomial

    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        denominator_all_zero = True
        for index, id in enumerate(np.ndindex(denominator.coefficients.shape)):
            if denominator.coefficients[id] != 0:
                denominator_all_zero = False

        if not denominator_all_zero:
            self.numerator = numerator
            self.denominator = denominator
        else:
            raise RuntimeError("dividing by 0 is not allowed")

    # TODO valuation function

    def __str__(self) -> str:
        s = str(self.numerator) + "\n"
        # the length of the division line depends on the length of the largest polynomial
        for i in range(max(len(str(self.numerator)), len(str(self.denominator)))):
            s += "-"
        s += "\n" + str(self.denominator)
        return s


if __name__ == "__main__":
    polynomial1 = Polynomial(3, 3)
    polynomial2 = Polynomial(4, 4)
    polynomial1.set_coefficient((0, 0, 1), 5)
    polynomial1.set_coefficient((3, 2, 1), 2.4)
    polynomial2.set_coefficient((2, 3, 2, 1), 1)
    polynomial2.set_coefficient((0, 0, 0, 0), 3)

    rationalfunction = RationalFunction(polynomial1, polynomial2)

    print(rationalfunction)
